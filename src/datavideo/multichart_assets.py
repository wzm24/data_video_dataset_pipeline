from __future__ import annotations

import cv2
import re
import shutil
from pathlib import Path
from typing import Any

from datavideo.frames import extract_frames, write_frame_manifest
from datavideo.keyframes import _clamp01, _image_motion_scores, extract_still
from datavideo.cv_align import (
    _detect_tick_label_blocks,
    bar_layout_regularity,
    detect_axis_tick_marks,
    detect_bars,
    detect_lines,
)
from datavideo.media import ffprobe
from datavideo.semantic import build_semantic_svg
from datavideo.schemas import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl
from datavideo.visual_provenance import assert_qwen_visual_inputs
from datavideo.dynamic_data import (
    build_dynamic_records,
    load_narration_sentences,
    plan_dynamic_state_keyframes,
    plan_state_sampling,
    write_dynamic_outputs,
)

from .multichart_qwen import MultichartQwenClient


def _clip_id(row: dict[str, Any]) -> str:
    return str(
        row.get("output_stem")
        or row.get("clip_id")
        or f"{row.get('chart_type') or 'chart'}_{row.get('chart_index') or 0}"
    )


def _keyframe_asset(keyframes: dict[str, Any]) -> Path:
    """Resolve the primary keyframe asset (upstream uses ``selected``, older
    manifests used ``initial``)."""
    assets = keyframes.get("assets") if isinstance(keyframes.get("assets"), dict) else {}
    for name in ("selected", "initial"):
        value = assets.get(name)
        if value:
            return Path(str(value))
    return Path("")


def _keyframe_timestamp(keyframes: dict[str, Any]) -> float | None:
    timestamps = keyframes.get("timestamps") if isinstance(keyframes.get("timestamps"), dict) else {}
    for name in ("selected", "initial"):
        value = timestamps.get(name)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _duration_seconds(video: str | Path) -> float:
    return float(ffprobe(video)["format"]["duration"])


def _safe_still_timestamp(timestamp: float, duration: float, cfg: dict[str, Any]) -> float:
    margin = float(cfg.get("keyframes", {}).get("tail_frame_min_margin_seconds", 0.25))
    return max(0.0, min(float(timestamp), max(0.0, duration - margin)))


def _sample_fps(cfg: dict[str, Any]) -> float:
    requested = cfg.get("keyframes", {}).get("sample_fps", cfg.get("sampling", {}).get("fine_fps", 4))
    try:
        return max(1.0, min(8.0, float(requested)))
    except (TypeError, ValueError):
        return 4.0


def _clip_context(row: dict[str, Any], duration: float) -> dict[str, Any]:
    return {
        "clip_id": _clip_id(row),
        "chart_type": row.get("chart_type"),
        "title": row.get("raw_video_title"),
        "channel": row.get("channel"),
        "year": row.get("year"),
        "source_youtube_url": row.get("youtube_url"),
        "source_time_range": {"start": row.get("start_seconds", row.get("start_time")), "end": row.get("end_seconds", row.get("end_time"))},
        "clip_duration_seconds": round(duration, 3),
        "video_role": "visual_clip",
    }


def _heuristic_keyframe_score(motion_score: float) -> dict[str, Any]:
    staticness = 1.0 - motion_score
    return {
        "target_chart_type_match": True,
        "same_chart": True,
        "scene_change_or_title_card": False,
        "scene_change": False,
        "structure_complete": True,
        "complete_chart": True,
        "final_or_most_complete_state": True,
        "data_marks_readable": True,
        "printed_text_readable": False,
        "labels_readable": False,
        "edge_crop_or_occlusion": False,
        "has_directly_printed_values": False,
        "staticness": staticness,
        "completeness": 0.65,
        "state_finality": 0.65,
        "edge_integrity": 1.0,
        "data_text_visibility": 0.5,
        "chart_identity_consistency": 0.75,
        "data_extraction_suitability": 0.55,
        "motion_score": motion_score,
        "state_summary": "heuristic fallback",
        "reason": "heuristic fallback from sampled clip frame and image motion",
    }


_CHART_TYPE_TERMS = {
    "map": {"map", "geographic", "region"},
    "bar": {"bar", "bars"},
    "line": {"line"},
    "area": {"area"},
    "donut": {"donut"},
    "pie": {"pie"},
    "timeline": {"timeline", "time line", "event", "events"},
    "treemap": {"treemap", "tree map", "rectangular", "rectangle", "segments"},
    "scatter": {"scatter", "points"},
    "sankey": {"sankey", "flow", "flows", "nodes"},
    "pictograph": {"pictograph", "icon", "icons"},
    "combined": {"combined"},
}


def _reason_type_penalty(score: dict[str, Any], chart_type: str) -> float:
    text = f"{score.get('reason', '')} {score.get('state_summary', '')}".lower()
    if not text:
        return 0.0
    target_terms = _CHART_TYPE_TERMS.get(chart_type, {chart_type})
    other_terms = set().union(*[terms for key, terms in _CHART_TYPE_TERMS.items() if key != chart_type])
    mentions_target = any(term in text for term in target_terms)
    mentions_other = any(term in text for term in other_terms)
    if mentions_other and not mentions_target:
        return 3.0
    return 0.0


def _combined_keyframe_score(score: dict[str, Any]) -> float:
    return (
        1.8 * _clamp01(score.get("completeness"))
        + 1.4 * _clamp01(score.get("state_finality"))
        + 1.2 * _clamp01(score.get("edge_integrity"))
        + 0.9 * _clamp01(score.get("data_text_visibility"))
        + 0.7 * _clamp01(score.get("chart_identity_consistency"))
        + (1.0 if score.get("target_chart_type_match", score.get("same_chart")) else -4.0)
        + (0.8 if score.get("structure_complete", score.get("complete_chart")) else -2.0)
        + (0.7 if score.get("final_or_most_complete_state") else -0.8)
        + (0.5 if score.get("data_marks_readable") else -1.0)
        + (0.4 if score.get("printed_text_readable", score.get("labels_readable")) else 0.0)
        + (0.3 if score.get("has_directly_printed_values") else 0.0)
        - (2.0 if score.get("edge_crop_or_occlusion") else 0.0)
        - (2.5 if score.get("scene_change_or_title_card", score.get("scene_change")) else 0.0)
        - 0.2 * _clamp01(score.get("motion_score"), default=1.0)
    )


def _image_sharpness(path: str | Path) -> float:
    """Laplacian variance as a blur measure.  Low values mean a blurry frame
    (motion blur / cross-fade), which is a poor static representative."""
    try:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return 0.0
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _state_enabled_chart_types(cfg: dict[str, Any]) -> set[str]:
    values = cfg.get("keyframes", {}).get("state_enabled_chart_types", [])
    return {str(value).lower() for value in _as_list(values)}


def _prefer_late_chart_types(cfg: dict[str, Any]) -> set[str]:
    values = cfg.get("keyframes", {}).get("prefer_late_chart_types", ["bar", "line", "pie", "donut", "combined"])
    return {str(value).lower() for value in _as_list(values)}


def _selection_rank(row: dict[str, Any], chart_type: str, cfg: dict[str, Any]) -> tuple[bool, bool, bool, bool, bool, float, float, bool, float, float]:
    score = row["score"]
    duration = float(row.get("clip_duration", 0.0))
    timestamp = float(row["timestamp"])
    time_position = timestamp / duration if duration else 0.0
    late_priority = time_position if chart_type.lower() in _prefer_late_chart_types(cfg) else 0.0
    return (
        bool(score.get("target_chart_type_match", score.get("same_chart"))),
        not bool(score.get("scene_change_or_title_card", score.get("scene_change"))),
        bool(score.get("structure_complete", score.get("complete_chart"))),
        not bool(score.get("edge_crop_or_occlusion")),
        bool(score.get("final_or_most_complete_state")),
        _clamp01(score.get("completeness")),
        _clamp01(score.get("state_finality")),
        bool(score.get("data_marks_readable")),
        late_priority,
        float(row["combined_score"]),
    )


def _add_tail_candidate_frames(
    normalized_video: str | Path,
    frame_dir: Path,
    frames: list[dict[str, Any]],
    duration: float,
    fps: float,
    cfg: dict[str, Any],
    force: bool,
) -> list[dict[str, Any]]:
    min_tail_margin = float(cfg.get("keyframes", {}).get("tail_frame_min_margin_seconds", 0.25))
    near_end_margin = min(0.05, max(0.0, min_tail_margin))
    offsets = [float(x) for x in (0.75, 0.5, min_tail_margin, near_end_margin)]
    existing_times = [float(frame["timestamp"]) for frame in frames]
    rows = list(frames)
    for idx, offset in enumerate(offsets, start=1):
        timestamp = max(0.0, min(duration - near_end_margin, duration - offset))
        if any(abs(timestamp - t) <= (0.5 / fps) for t in existing_times):
            continue
        path = frame_dir / f"keyframe_candidate_tail_{idx:02d}.jpg"
        extract_still(normalized_video, timestamp, path, force=force)
        rows.append(
            {
                "frame_id": path.stem,
                "path": str(path),
                "timestamp": timestamp,
                "fps": fps,
                "sample_type": "keyframe_candidate_tail",
            }
        )
        existing_times.append(timestamp)
    return sorted(rows, key=lambda item: float(item["timestamp"]))


def _quality_rows(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in scored_rows:
        score = row["score"]
        if not score.get("target_chart_type_match", score.get("same_chart")):
            continue
        if score.get("scene_change_or_title_card", score.get("scene_change")):
            continue
        if not score.get("structure_complete", score.get("complete_chart")):
            continue
        if score.get("edge_crop_or_occlusion"):
            continue
        rows.append(row)
    return rows or scored_rows


def _pick_evenly(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    if limit <= 1:
        return [rows[-1]]
    selected = []
    for idx in range(limit):
        pos = round(idx * (len(rows) - 1) / (limit - 1))
        selected.append(rows[pos])
    seen = set()
    unique = []
    for row in selected:
        if row["frame_id"] not in seen:
            unique.append(row)
            seen.add(row["frame_id"])
    return unique


def _select_state_rows(scored_rows: list[dict[str, Any]], primary: dict[str, Any], cfg: dict[str, Any], chart_type: str) -> list[dict[str, Any]]:
    if chart_type.lower() not in _state_enabled_chart_types(cfg):
        return []
    max_states = int(cfg.get("keyframes", {}).get("max_state_keyframes", 3))
    if max_states <= 0:
        return []
    rows = sorted(_quality_rows(scored_rows), key=lambda item: float(item["timestamp"]))
    if len(rows) <= 1:
        return [primary]
    candidates = _pick_evenly(rows, max_states)
    if primary["frame_id"] not in {row["frame_id"] for row in candidates}:
        others = [row for row in candidates if row["frame_id"] != primary["frame_id"]]
        candidates = _pick_evenly(sorted(others, key=lambda item: float(item["timestamp"])), max_states - 1) + [primary]
    return sorted(candidates, key=lambda item: float(item["timestamp"]))


def _select_clip_data_rows(scored_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    max_frames = int(cfg.get("clip_data", {}).get("max_frames", 10))
    rows = sorted(_quality_rows(scored_rows), key=lambda item: float(item["timestamp"]))
    selected = _pick_evenly(rows, max(1, max_frames))
    by_id = {row["frame_id"]: row for row in selected}
    for row in state_rows:
        by_id[row["frame_id"]] = row
    merged = sorted(by_id.values(), key=lambda item: float(item["timestamp"]))
    return _pick_evenly(merged, max_frames)


def _clip_data_row_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("year"),
        row.get("state"),
        row.get("label"),
        row.get("series"),
        row.get("x"),
        row.get("y"),
        row.get("value"),
        row.get("unit"),
        row.get("raw_text"),
        row.get("source_frame"),
        row.get("time_seconds"),
    )


def _merge_clip_data(main: dict[str, Any], supplemental: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {**main}
    rows = list(main.get("rows") if isinstance(main.get("rows"), list) else [])
    seen = {_clip_data_row_key(row) for row in rows if isinstance(row, dict)}
    visible_text = list(main.get("visible_text") if isinstance(main.get("visible_text"), list) else [])
    uncertain_fields = list(main.get("uncertain_fields") if isinstance(main.get("uncertain_fields"), list) else [])
    manual_rows = list(main.get("manual_stub_rows") if isinstance(main.get("manual_stub_rows"), list) else [])
    for data in supplemental:
        if not isinstance(data, dict):
            continue
        for row in data.get("rows") if isinstance(data.get("rows"), list) else []:
            if not isinstance(row, dict):
                continue
            key = _clip_data_row_key(row)
            if key in seen:
                continue
            rows.append(row)
            seen.add(key)
        for text in data.get("visible_text") if isinstance(data.get("visible_text"), list) else []:
            if text not in visible_text:
                visible_text.append(text)
        for field in data.get("uncertain_fields") if isinstance(data.get("uncertain_fields"), list) else []:
            if field not in uncertain_fields:
                uncertain_fields.append(field)
        for row in data.get("manual_stub_rows") if isinstance(data.get("manual_stub_rows"), list) else []:
            if row not in manual_rows:
                manual_rows.append(row)
        if not merged.get("title") and data.get("title"):
            merged["title"] = data.get("title")
        if not merged.get("unit") and data.get("unit"):
            merged["unit"] = data.get("unit")
        if not merged.get("x_axis") and data.get("x_axis"):
            merged["x_axis"] = data.get("x_axis")
        if not merged.get("y_axis") and data.get("y_axis"):
            merged["y_axis"] = data.get("y_axis")
    merged["rows"] = rows
    merged["visible_text"] = visible_text
    merged["uncertain_fields"] = uncertain_fields
    merged["manual_stub_rows"] = manual_rows
    merged["has_extractable_data"] = bool(rows)
    merged["needs_manual_data"] = bool(merged.get("needs_manual_data")) or bool(manual_rows)
    merged["temporal_change"] = bool(merged.get("temporal_change")) or len({(row.get("year"), row.get("state")) for row in rows if isinstance(row, dict) and (row.get("year") not in (None, "") or row.get("state") not in (None, ""))}) > 1
    return merged


def _anchor_frame_indices(frame_context: list[dict[str, Any]], cfg: dict[str, Any]) -> list[int]:
    max_anchor_frames = int(cfg.get("clip_data", {}).get("supplemental_anchor_frames", 2))
    if max_anchor_frames <= 0 or len(frame_context) <= 1:
        return []
    ordered = sorted(range(len(frame_context)), key=lambda idx: float(frame_context[idx].get("time_seconds") or 0.0))
    anchors = [ordered[0], ordered[-1]]
    if max_anchor_frames > 2:
        midpoint = ordered[len(ordered) // 2]
        anchors.insert(1, midpoint)
    unique = []
    for idx in anchors:
        if idx not in unique:
            unique.append(idx)
    return unique[:max_anchor_frames]


def _metadata_entities(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = data.get("rows") if isinstance(data, dict) else []
    entities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or row.get("value") in (None, ""):
            continue
        label = row.get("label") or row.get("series") or row.get("x") or row.get("state")
        if not label:
            label = f"entity_{len(entities) + 1}"
        key = str(label).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            {
                "label": str(label).strip(),
                "value": row.get("value"),
                "unit": row.get("unit"),
            }
        )
    return entities


def _write_semantic_state_inputs(
    dynamic: dict[str, Any],
    source_video: str | Path | None,
    out_dir: str | Path,
    cfg: dict[str, Any],
    *,
    force: bool,
) -> dict[str, Any]:
    plan = plan_dynamic_state_keyframes(
        dynamic,
        max_states=int(cfg.get("dynamic_data", {}).get("max_state_keyframes", 8) or 8),
    )
    manifest_path = Path(out_dir) / "semantic_state_input_manifest.json"
    state_dir = ensure_dir(Path(out_dir) / "keyframes" / "states")
    if not plan.get("should_save"):
        manifest = {"should_save": False, "reason": plan.get("reason"), "semantic_inputs": []}
        write_json(manifest_path, manifest)
        return {"manifest": str(manifest_path), "semantic_inputs": []}

    source_path = Path(source_video) if source_video else None
    rows = []
    if force:
        for stale in state_dir.glob("state_*.png"):
            stale.unlink()
    for idx, state in enumerate(plan.get("states", []), start=1):
        label = str(state.get("state_label") or state.get("state_key") or state.get("state_id") or idx)
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_") or str(idx)
        out_path = state_dir / f"state_{idx:03d}_{safe_label}.png"
        timestamp = state.get("timestamp")
        if source_path and source_path.exists() and timestamp is not None:
            try:
                asset = extract_still(source_path, float(timestamp), out_path, force=force)
            except Exception:
                asset = None
            if asset is None or not Path(asset).exists() or Path(asset).stat().st_size == 0:
                asset = None
        else:
            asset = None
        if asset is None:
            source_frame = Path(state.get("source_frame_path") or "")
            if source_frame.exists():
                asset = out_path
                try:
                    # The source frame may be a JPEG/other format while the
                    # semantic pipeline expects a real PNG payload (it
                    # inspects the file signature, not just the extension).
                    from PIL import Image

                    Image.open(source_frame).convert("RGB").save(asset, "PNG")
                except Exception:
                    asset = None
        if asset is None:
            rows.append(
                {
                    **state,
                    "keyframe": None,
                    "semantic_input": None,
                    "success": False,
                    "failure_reason": "state keyframe extraction failed",
                }
            )
            continue
        rows.append({**state, "keyframe": str(asset), "semantic_input": str(asset)})

    manifest = {
        "should_save": True,
        "reason": plan.get("reason"),
        "selection_rule": "all_complete_evidenced_data_states_as_state_keyframes",
        "semantic_inputs": rows,
    }
    write_json(manifest_path, manifest)
    _merge_dynamic_state_keyframes(Path(out_dir), rows)
    return {"manifest": str(manifest_path), "semantic_inputs": rows}


def _merge_dynamic_state_keyframes(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path = out_dir / "keyframes" / "keyframe_manifest.json"
    if not manifest_path.exists():
        return
    manifest = read_json(manifest_path)
    states = []
    for idx, item in enumerate(rows, start=1):
        asset = item.get("keyframe") or item.get("semantic_input")
        if not asset:
            continue
        state_id = str(item.get("state_id") or f"state_{idx:03d}")
        state_label = str(item.get("state_label") or item.get("state_key") or state_id)
        states.append(
            {
                "name": state_id,
                "state_key": item.get("state_key"),
                "state_label": state_label,
                "timestamp": item.get("timestamp"),
                "asset": asset,
                "source_frame_id": item.get("source_frame_id"),
                "source_frame_path": item.get("source_frame_path"),
                "keyframe_role": "data_state_keyframe",
                "entity_ids": item.get("entity_ids", []),
                "signature": item.get("signature", []),
            }
        )
    manifest.setdefault("assets", {})["states"] = [state["asset"] for state in states]
    manifest["states"] = states
    manifest["state_keyframe_selection_method"] = "dynamic_data_first_last_state_keyframes"
    write_json(manifest_path, manifest)


def build_semantic_state_svgs(
    semantic_state_inputs: dict[str, Any] | None,
    out_dir: str | Path,
    cfg: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    manifest_path = out_dir / "semantic_state_svg_manifest.json"
    inputs = (semantic_state_inputs or {}).get("semantic_inputs")
    if not isinstance(inputs, list) or not inputs:
        manifest = {"should_save": False, "reason": (semantic_state_inputs or {}).get("reason", "no_semantic_state_inputs"), "semantic_svgs": []}
        write_json(manifest_path, manifest)
        return manifest

    rows = []
    for item in inputs:
        if not isinstance(item, dict):
            continue
        image_path = Path(item.get("semantic_input") or "")
        try:
            image_path = _ensure_png_payload(image_path)
        except Exception:
            pass
        if not image_path.exists():
            rows.append({**item, "success": False, "failure_reason": "semantic input image missing"})
            continue
        state_id = str(item.get("state_id") or f"state_{len(rows) + 1:03d}")
        state_label = str(item.get("state_label") or item.get("state_key") or state_id)
        state_key = str(item.get("state_key") or state_label or state_id)
        safe_key = re.sub(r"[^a-z0-9]+", "-", state_key.lower()).strip("-") or state_id
        state_out_dir = ensure_dir(out_dir / "semantic_states" / safe_key / "vision")
        parent_metadata = out_dir / "chart_metadata.json"
        if parent_metadata.exists():
            shutil.copy2(parent_metadata, state_out_dir / "chart_metadata.json")
        report = build_semantic_svg(image_path, state_out_dir, cfg, force=force, rebuild_components=force)
        rows.append({**item, **report})

    manifest = {
        "should_save": bool(rows),
        "reason": "semantic_state_inputs_converted" if rows else "no_valid_semantic_state_inputs",
        "semantic_svgs": rows,
    }
    write_json(manifest_path, manifest)
    return manifest


def _ensure_png_payload(path: Path) -> Path:
    """Repair a .png file whose payload is not PNG (e.g. a JPEG still written
    to a .png path by an older ffmpeg invocation or a copied source frame).

    The semantic pipeline inspects the file signature (not just the
    extension), so such files make every downstream step fail with "Not a PNG
    image". Re-encode them in place with Pillow.
    """
    if not path.exists() or path.suffix.lower() != ".png":
        return path
    header = path.read_bytes()[:8]
    if header == b"\x89PNG\r\n\x1a\n":
        return path
    from PIL import Image

    Image.open(path).convert("RGB").save(path, "PNG")
    return path


def _source_fps(video: str | Path) -> float | None:
    try:
        streams = ffprobe(video).get("streams", [])
    except Exception:
        return None
    for stream in streams:
        if stream.get("codec_type") != "video":
            continue
        rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "")
        if "/" in rate:
            num, den = rate.split("/", 1)
            try:
                fps = float(num) / float(den)
            except (TypeError, ValueError, ZeroDivisionError):
                continue
        else:
            try:
                fps = float(rate)
            except (TypeError, ValueError):
                continue
        if fps > 0:
            return fps
    return None


def _extract_state_evidence_rows(
    video: str | Path,
    clip_id: str,
    sampling_plan: dict[str, Any],
    cfg: dict[str, Any],
    *,
    force: bool,
) -> list[dict[str, Any]]:
    windows = sampling_plan.get("fine_windows") if isinstance(sampling_plan.get("fine_windows"), list) else []
    if not windows:
        return []
    clip_cfg = cfg.get("clip_data", {})
    short_side = int(cfg.get("sampling", {}).get("short_side", 768))
    min_window_frames = int(clip_cfg.get("min_fine_window_frames", 2))
    max_source_fps = float(clip_cfg.get("max_source_fps", 30))
    evidence_dir = ensure_dir(Path(cfg.get("processed_root", "data/processed")) / clip_id / "visual_frames" / "state_evidence")
    rows: list[dict[str, Any]] = []
    for idx, window in enumerate(windows, start=1):
        start = max(0.0, float(window.get("start", 0.0)))
        end = max(start, float(window.get("end", start)))
        fine_fps = max(1.0, float(window.get("target_fps") or sampling_plan.get("fine_fps") or 8.0))
        prefix = f"state_evidence_w{idx:03d}_{int(round(fine_fps))}fps"
        fine_rows = extract_frames(video, evidence_dir, fine_fps, short_side, prefix, force=force, start=start, end=end)
        for row in fine_rows:
            row["score"] = {"motion_score": 0.0, "state_summary": "local dynamic data evidence frame"}
            row["combined_score"] = 0.0
            row["sample_type"] = "state_evidence"
            row["sampling_reason"] = window.get("reason", "fine_window")
        rows.extend(fine_rows)
        if len(fine_rows) < min_window_frames:
            source_fps = _source_fps(video)
            if source_fps and source_fps > fine_fps:
                source_prefix = f"state_evidence_w{idx:03d}_sourcefps"
                source_rows = extract_frames(
                    video,
                    evidence_dir,
                    min(source_fps, max_source_fps),
                    short_side,
                    source_prefix,
                    force=force,
                    start=start,
                    end=end,
                )
                for row in source_rows:
                    row["score"] = {"motion_score": 0.0, "state_summary": "source-fps dynamic data evidence frame"}
                    row["combined_score"] = 0.0
                    row["sample_type"] = "state_evidence_source_fps"
                    row["sampling_reason"] = "fine_window_insufficient"
                rows.extend(source_rows)
    by_time: dict[float, dict[str, Any]] = {}
    for row in rows:
        by_time[round(float(row["timestamp"]), 3)] = row
    return [by_time[key] for key in sorted(by_time)]


def select_keyframe(
    normalized_video: str | Path,
    row: dict[str, Any],
    out_dir: str | Path,
    cfg: dict[str, Any],
    *,
    client: MultichartQwenClient | None = None,
    force: bool = False,
    context_video: str | Path | None = None,
    context_visual_end: float | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    manifest_path = out_dir / "keyframe_manifest.json"
    clip_id = _clip_id(row)
    if manifest_path.exists() and not force:
        cached = read_json(manifest_path)
        selected = _keyframe_asset(cached)
        if cached.get("clip_id") == clip_id and selected.exists():
            return cached

    duration = _duration_seconds(normalized_video)
    frame_dir = ensure_dir(Path(cfg.get("processed_root", "data/processed")) / clip_id / "visual_frames" / "keyframe_candidates")
    fps = _sample_fps(cfg)
    frames = extract_frames(
        normalized_video,
        frame_dir,
        fps,
        int(cfg.get("sampling", {}).get("short_side", 768)),
        "keyframe_candidate",
        force=force,
    )
    frames = _add_tail_candidate_frames(normalized_video, frame_dir, frames, duration, fps, cfg, force)
    if not frames:
        raise RuntimeError(f"No keyframe candidates sampled for {clip_id}")
    write_frame_manifest(frame_dir.parent / "keyframe_frame_manifest.jsonl", frames)
    assert_qwen_visual_inputs([frame["path"] for frame in frames])

    motion_scores = _image_motion_scores(frames)
    scorer = client or MultichartQwenClient(cfg)
    context = _clip_context(row, duration)
    scored_rows = []
    for frame in frames:
        motion_score = motion_scores.get(frame["frame_id"], 0.0)
        model_score = scorer.score_keyframe_candidate(frame["path"], context)
        score = model_score["result"]
        if model_score["model_status"] != "qwen":
            score = _heuristic_keyframe_score(motion_score)
            score["target_chart_type_match"] = False
            score["same_chart"] = False
            score["structure_complete"] = False
            score["complete_chart"] = False
            score["state_finality"] = 0.0
            score["reason"] = model_score["failure_reason"] or score["reason"]
        else:
            score["motion_score"] = max(_clamp01(score.get("motion_score"), default=1.0), motion_score)
        combined = _combined_keyframe_score(score) - _reason_type_penalty(score, str(row.get("chart_type", "")))
        scored_rows.append(
            {
                **frame,
                "clip_duration": duration,
                "score": score,
                "combined_score": round(combined, 4),
                "raw_response": model_score["raw_response"],
                "model_status": model_score["model_status"],
                "failure_reason": model_score["failure_reason"],
                "sharpness": _image_sharpness(frame["path"]),
            }
        )

    chart_type = str(row.get("chart_type", ""))
    # Deterministic completeness signal for bar-like charts: the more bars a
    # candidate frame shows, the more complete it is.  Qwen sometimes marks a
    # mid-animation frame as "complete"; the CV bar count re-ranks in favour
    # of the frame that actually has the most bars.
    if "bar" in chart_type or "combined" in chart_type:
        for item in scored_rows:
            try:
                bars = detect_bars(item["path"])
                item["cv_bar_count"] = len(bars)
                orientation = bars[0].get("orientation") if bars else "vertical"
                item["cv_tick_count"] = len(detect_axis_tick_marks(item["path"], orientation))
                item["bar_regularity"] = bar_layout_regularity(bars)
            except Exception:
                item["cv_bar_count"] = 0
                item["cv_tick_count"] = 0
                item["bar_regularity"] = 0.0
            # A frame with more bars is more complete, a frame with the
            # value-axis tick marks drawn is far more usable for value
            # estimation, and a regular bar layout excludes cross-fade frames
            # where two charts are superimposed.
            item["combined_score"] = round(
                item["combined_score"]
                + 0.4 * item["cv_bar_count"]
                + 0.25 * item["cv_tick_count"]
                - 3.0 * (1.0 - item["bar_regularity"]),
                4,
            )
        # The clip-wide full bar count is the modal count among frames that
        # show at least two bars.  A single noisy over-count (text blob) or
        # an incomplete early frame must not define the target.
        counts = [
            int(item["cv_bar_count"])
            for item in scored_rows
            if int(item["cv_bar_count"]) >= 2
        ]
        modal_count = max(set(counts), key=counts.count) if counts else 0
        for item in scored_rows:
            item["_bar_full_count"] = modal_count
    # Blurry frames (motion blur, cross-fade, mid-transition) are poor static
    # representatives regardless of chart type.  Penalise frames that are far
    # below the clip's own sharpness median, so the score is scale-agnostic.
    sharpness_values = [item.get("sharpness", 0.0) for item in scored_rows]
    med_sharp = float(sorted(sharpness_values)[len(sharpness_values) // 2]) if sharpness_values else 0.0
    if med_sharp > 0:
        # Bar charts: blurry frames (motion blur / cross-fade) are excluded
        # outright when a sharp alternative exists, so the "most bars" rule
        # never lands on a smeared transition frame.
        is_bar_like = "bar" in chart_type or "combined" in chart_type
        clear = [item for item in scored_rows if item.get("sharpness", 0.0) >= 0.35 * med_sharp]
        if is_bar_like and len(clear) >= 1:
            for item in scored_rows:
                if item.get("sharpness", 0.0) < 0.35 * med_sharp:
                    item["blur_penalized"] = True
            scored_rows = clear
        else:
            for item in scored_rows:
                if item.get("sharpness", 0.0) < 0.35 * med_sharp:
                    item["combined_score"] = round(item["combined_score"] - 6.0, 4)
                    item["blur_penalized"] = True
    # Line/area charts draw both the polylines and the axis ticks
    # progressively: completeness is the drawing progress (rightmost point
    # of the slowest line, normalised by frame width) plus how many value
    # ticks have been drawn (a frame with a complete axis is usable for value
    # estimation). Point counts and the raw number of detected lines are NOT
    # used: illustrations / title cards / annotation arrows produce spurious
    # lines and points, and finished vs near-finished frames often have
    # identical counts.
    if "line" in chart_type or "area" in chart_type or "timeline" in chart_type:
        for item in scored_rows:
            try:
                detected_lines = detect_lines(item["path"])
                frame = cv2.imread(str(item["path"]))
                frame_w = frame.shape[1] if frame is not None else 1
                has_value_axis = bool(
                    frame is not None and len(_detect_tick_label_blocks(frame, "vertical")) >= 2
                )
                item["line_point_count"] = (
                    sum(len(line.get("points", [])) for line in detected_lines)
                    if has_value_axis
                    else 0
                )
                item["line_progress"] = 0.0
                item["line_tick_count"] = 0
                if has_value_axis and detected_lines:
                    rightmost = [
                        max(px for px, _ in line.get("points", []))
                        for line in detected_lines
                        if line.get("points")
                    ]
                    if rightmost:
                        item["line_progress"] = round(min(rightmost) / frame_w, 4)
                    try:
                        item["line_tick_count"] = len(detect_axis_tick_marks(item["path"], "vertical"))
                    except Exception:
                        item["line_tick_count"] = 0
            except Exception:
                item["line_point_count"] = 0
                item["line_progress"] = 0.0
                item["line_tick_count"] = 0
            item["combined_score"] = round(
                item["combined_score"]
                + 8.0 * item["line_progress"]
                + 1.5 * item["line_tick_count"],
                4,
            )
    selected = max(scored_rows, key=lambda item: _selection_rank(item, chart_type, cfg))
    keyframe_source_role = "visual_clip"
    boundary_reason = ""
    needs_review = False
    # When the annotated interval ends mid-animation, the complete chart may
    # only appear in the padded context tail.  For bar-like charts we scan a
    # short window right after the visual interval and use the frame with the
    # most bars; the clip boundary is flagged for review.
    if (
        context_video
        and Path(context_video).exists()
        and context_visual_end is not None
        and ("bar" in chart_type or "combined" in chart_type)
    ):
        try:
            ctx_duration = _duration_seconds(context_video)
            # Use the clip-wide modal full count as the tail gate, not the
            # noisy per-frame maximum (a single over-counted frame would
            # otherwise raise the bar and block a genuinely complete tail).
            best_clip_bars = max(
                (int(item.get("_bar_full_count") or 0) for item in scored_rows),
                default=0,
            )
            if best_clip_bars <= 0:
                best_clip_bars = max((item.get("cv_bar_count") or 0) for item in scored_rows)
            scan_dir = ensure_dir(Path(cfg.get("processed_root", "data/processed")) / clip_id / "context_tail_frames")
            # Take the first frame *after* the annotated interval that is more
            # complete than anything inside it.  This catches charts that
            # finish right after the cut, while avoiding far-away transition
            # frames (where noisy components inflate the count).
            chosen_tail = None
            t = float(context_visual_end) + 0.1
            tail_limit = min(ctx_duration - 0.05, float(context_visual_end) + 2.0)
            while t < tail_limit and chosen_tail is None:
                path = scan_dir / f"context_tail_{t:.2f}.png"
                try:
                    extract_still(context_video, t, path, force=True)
                    tail_sharp = _image_sharpness(str(path))
                    if med_sharp > 0 and tail_sharp < 0.35 * med_sharp:
                        t += 0.3
                        continue
                    tail_bars = detect_bars(str(path))
                    cnt = len(tail_bars)
                    tail_orientation = tail_bars[0].get("orientation") if tail_bars else "vertical"
                    tick_cnt = len(detect_axis_tick_marks(str(path), tail_orientation))
                    regularity = bar_layout_regularity(tail_bars)
                except Exception:
                    cnt = 0
                    tick_cnt = 0
                    regularity = 0.0
                # The tail frame must be at least as complete as the best clip
                # frame *and* still carry the axis tick marks; a frame where
                # the bars have grown but the grid lines already faded out is
                # not usable for tick-based estimation, and a cross-fade frame
                # (irregular bar layout) must never be selected.
                if cnt >= best_clip_bars and tick_cnt >= 2 and regularity >= 0.7:
                    chosen_tail = (t, cnt, str(path))
                t += 0.3
            if chosen_tail:
                t, cnt, path = chosen_tail
                clip_rel_ts = min(round(float(t) - float(context_visual_end) + duration, 3), max(0.0, duration - 0.05))
                selected = {
                    "timestamp": clip_rel_ts,
                    "context_tail_timestamp": round(float(t), 3),
                    "path": path,
                    "frame_id": Path(path).stem,
                    "clip_duration": duration,
                    "score": {
                        "target_chart_type_match": True,
                        "same_chart": True,
                        "scene_change_or_title_card": False,
                        "scene_change": False,
                        "structure_complete": True,
                        "complete_chart": True,
                        "final_or_most_complete_state": True,
                        "data_marks_readable": True,
                        "printed_text_readable": True,
                        "labels_readable": True,
                        "edge_crop_or_occlusion": False,
                        "has_directly_printed_values": True,
                        "completeness": 1.0,
                        "state_finality": 1.0,
                        "data_text_visibility": 1.0,
                        "chart_identity_consistency": 1.0,
                        "motion_score": 0.0,
                        "state_summary": "context-tail frame with the most bars (CV verified)",
                        "reason": "selected by CV bar-count completeness scan of the context tail",
                    },
                    "combined_score": round(10.0 + cnt, 4),
                    "raw_response": None,
                    "model_status": "cv",
                    "failure_reason": None,
                    "cv_bar_count": cnt,
                }
                keyframe_source_role = "context_tail"
                boundary_reason = "boundary_mid_animation"
                needs_review = True
        except Exception:
            pass
    if keyframe_source_role == "context_tail":
        timestamp = float(selected["timestamp"])
        context_ts = float(selected["context_tail_timestamp"])
        asset = str(extract_still(context_video, context_ts, out_dir / "selected.png", force=True))
        state_rows: list[dict[str, Any]] = []
    else:
        # Verify the chosen frame at the final (full-resolution) extraction:
        # the scoring-time bar count comes from scaled candidate frames and
        # can differ (e.g. an early frame that detected text blobs as bars
        # but has no chart at full resolution).  Walk the ranked candidates
        # until one actually yields bars.
        ranked = sorted(
            scored_rows,
            key=lambda item: _selection_rank(item, chart_type, cfg),
            reverse=True,
        )
        asset = None
        for candidate in ranked:
            candidate_ts = _safe_still_timestamp(float(candidate["timestamp"]), duration, cfg)
            probe_path = out_dir / "keyframe_probe.png"
            try:
                extract_still(normalized_video, candidate_ts, probe_path, force=True)
                probe_bars = detect_bars(str(probe_path))
            except Exception:
                probe_bars = []
            if len(probe_bars) >= 2:
                selected = candidate
                timestamp = candidate_ts
                asset = str(extract_still(normalized_video, timestamp, out_dir / "selected.png", force=True))
                break
        if asset is None:
            timestamp = _safe_still_timestamp(float(selected["timestamp"]), duration, cfg)
            asset = str(extract_still(normalized_video, timestamp, out_dir / "selected.png", force=True))
        state_rows = _select_state_rows(scored_rows, selected, cfg, chart_type)
    states_dir = ensure_dir(out_dir / "states")
    if force:
        for stale_state in states_dir.glob("state_*.png"):
            stale_state.unlink()
    states = []
    for idx, state_row in enumerate(state_rows, start=1):
        state_name = f"state_{idx:03d}"
        state_timestamp = _safe_still_timestamp(float(state_row["timestamp"]), duration, cfg)
        state_path = str(extract_still(normalized_video, state_timestamp, states_dir / f"{state_name}.png", force=True))
        states.append(
            {
                "name": state_name,
                "timestamp": state_timestamp,
                "asset": state_path,
                "source_frame_id": state_row["frame_id"],
                "combined_score": state_row["combined_score"],
                "state_summary": state_row["score"].get("state_summary"),
                "reason": state_row["score"].get("reason"),
            }
        )
    manifest = {
        "clip_id": clip_id,
        "chart_type": row["chart_type"],
        "timestamps": {"selected": timestamp},
        "context_tail_timestamp": selected.get("context_tail_timestamp"),
        "assets": {"selected": asset, "states": [state["asset"] for state in states]},
        "states": states,
        "selection_method": "v2_simple_complete_final_state_keyframe",
        "source_video_role": keyframe_source_role,
        "source_video": str(context_video if keyframe_source_role == "context_tail" else normalized_video),
        "boundary_reason": boundary_reason,
        "needs_review": needs_review,
        "source_frame_id": selected["frame_id"],
        "sample_fps": fps,
        "clip_context": context,
        "selected_score": selected["score"],
        "combined_score": selected["combined_score"],
        "score_manifest": str(out_dir / "keyframe_scores.jsonl"),
        "requirements": {
            "same_chart": True,
            "scene_change": False,
            "complete_chart": True,
            "data_marks_readable": True,
            "edge_crop_or_occlusion": False,
            "final_or_most_complete_state": True,
            "description": "Select a target-type frame with complete visible structure, no important edge crop, and a final or most complete chart state.",
        },
    }
    write_jsonl(out_dir / "keyframe_scores.jsonl", scored_rows)
    write_json(manifest_path, manifest)
    return manifest


def recover_clip_data(
    cfg: dict[str, Any],
    keyframes: dict[str, Any],
    row: dict[str, Any],
    out_dir: str | Path,
    *,
    client: MultichartQwenClient | None = None,
    force: bool = False,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    raw_path = out_dir / "chart_data_clip_raw.json"
    validation_path = out_dir / "chart_data_validation.json"
    csv_path = out_dir / "chart_data.csv"
    dynamic_json_path = out_dir / "dynamic_data.json"
    dynamic_csv_path = out_dir / "dynamic_data.csv"
    final_data_table_path = out_dir / "final_data_table.csv"
    data_change_events_path = out_dir / "data_change_events.csv"
    semantic_state_input_path = out_dir / "semantic_state_input_manifest.json"
    semantic_state_svg_path = out_dir / "semantic_state_svg_manifest.json"
    manual_csv_path = out_dir / "manual_data_stub.csv"
    events_path = out_dir / "data_events.jsonl"
    if raw_path.exists() and validation_path.exists() and not force:
        raw = read_json(raw_path)
        validation = read_json(validation_path)
        return {
            "data": raw.get("response", {}).get("data"),
            "dynamic_data": raw.get("dynamic_data", {}),
            "metadata": raw.get("metadata", {}),
            "validation": validation,
            "csv_path": str(csv_path) if csv_path.exists() else None,
            "dynamic_data_json": str(dynamic_json_path) if dynamic_json_path.exists() else None,
            "dynamic_data_csv": str(dynamic_csv_path) if dynamic_csv_path.exists() else None,
            "final_data_table_csv": str(final_data_table_path) if final_data_table_path.exists() else None,
            "data_change_events_csv": str(data_change_events_path) if data_change_events_path.exists() else None,
            "semantic_state_inputs": read_json(semantic_state_input_path) if semantic_state_input_path.exists() else None,
            "semantic_state_svgs": read_json(semantic_state_svg_path) if semantic_state_svg_path.exists() else None,
            "manual_csv_path": str(manual_csv_path) if manual_csv_path.exists() else None,
            "events_path": str(events_path) if events_path.exists() else None,
        }

    scorer = client or MultichartQwenClient(cfg)
    score_manifest = Path(keyframes.get("score_manifest", ""))
    scored_rows = read_jsonl(score_manifest) if score_manifest.exists() else []
    sampling_plan = plan_state_sampling(scored_rows, cfg) if scored_rows else {
        "coarse_fps": float(cfg.get("clip_data", {}).get("coarse_fps", 2)),
        "fine_fps": float(cfg.get("clip_data", {}).get("fine_fps", 8)),
        "coarse_frame_count": 0,
        "fine_windows": [],
        "selected_rows": [],
        "selected_frame_count": 0,
    }
    evidence_rows: list[dict[str, Any]] = []
    source_video = keyframes.get("source_video")
    if source_video and Path(source_video).exists():
        evidence_rows = _extract_state_evidence_rows(source_video, _clip_id(row), sampling_plan, cfg, force=force)
        if evidence_rows:
            write_frame_manifest(Path(cfg.get("processed_root", "data/processed")) / _clip_id(row) / "visual_frames" / "state_evidence_manifest.jsonl", evidence_rows)
    planned_rows = [*sampling_plan["selected_rows"], *evidence_rows]
    frame_rows = _select_clip_data_rows(planned_rows, scored_rows[:0], cfg) if planned_rows else []
    state_assets = {Path(state.get("asset", "")).resolve(): state for state in keyframes.get("states", [])}
    image_paths = []
    frame_context = []
    for idx, frame in enumerate(frame_rows, start=1):
        image_paths.append(frame["path"])
        frame_context.append(
            {
                "image_index": idx,
                "source_frame": frame["frame_id"],
                "time_seconds": round(float(frame["timestamp"]), 3),
                "state_summary": frame.get("score", {}).get("state_summary"),
            }
        )
    for state_path, state in state_assets.items():
        if str(state_path) not in {str(Path(path).resolve()) for path in image_paths} and state_path.exists():
            image_paths.append(str(state_path))
            frame_context.append(
                {
                    "image_index": len(image_paths),
                    "source_frame": state.get("source_frame_id"),
                    "time_seconds": round(float(state.get("timestamp", 0.0)), 3),
                    "state_summary": state.get("state_summary"),
                }
            )
    max_frames = int(cfg.get("clip_data", {}).get("max_frames", 10))
    if len(image_paths) > max_frames:
        paired = list(zip(image_paths, frame_context))
        paired = _pick_evenly(
            [{"path": path, "frame_context": context, "timestamp": context.get("time_seconds") or 0.0, "frame_id": str(idx)} for idx, (path, context) in enumerate(paired)],
            max_frames,
        )
        image_paths = [row["path"] for row in paired]
        frame_context = [row["frame_context"] for row in paired]
        for idx, context in enumerate(frame_context, start=1):
            context["image_index"] = idx
    if not image_paths:
        initial = _keyframe_asset(keyframes)
        if initial.exists():
            image_paths = [str(initial)]
            frame_context = [
                {
                    "image_index": 1,
                    "source_frame": keyframes.get("source_frame_id"),
                    "time_seconds": _keyframe_timestamp(keyframes),
                }
            ]
    if not image_paths:
        raise RuntimeError(f"No frames available for clip-level data recovery for {_clip_id(row)}")
    assert_qwen_visual_inputs(image_paths)

    context = {
        "clip_id": _clip_id(row),
        "chart_type": row.get("chart_type"),
        "title": row.get("raw_video_title"),
        "channel": row.get("channel"),
        "year": row.get("year"),
        "source_time_range": {"start": row.get("start_seconds", row.get("start_time")), "end": row.get("end_seconds", row.get("end_time"))},
        "video_role": "visual_clip",
        "rule": "Recover only directly printed values from the whole clip frame sequence; do not estimate geometric values.",
    }
    response = scorer.recover_clip_data(image_paths, context, frame_context)
    supplemental_responses = []
    supplemental_data = []
    for frame_idx in _anchor_frame_indices(frame_context, cfg):
        anchor_image = image_paths[frame_idx]
        anchor_context = {**frame_context[frame_idx], "image_index": 1}
        try:
            supplemental_response = scorer.recover_clip_data(
                [anchor_image],
                {
                    **context,
                    "rule": "Recover directly printed values from this single representative state frame; keep any printed year/state visible in the frame.",
                },
                [anchor_context],
            )
        except Exception as exc:
            supplemental_response = {
                "data": {},
                "raw_response": None,
                "model_status": "failed",
                "failure_reason": f"supplemental anchor recovery failed: {exc}",
            }
        supplemental_responses.append({"frame_context": anchor_context, **supplemental_response})
        if isinstance(supplemental_response.get("data"), dict) and supplemental_response.get("model_status") == "qwen":
            supplemental_data.append(supplemental_response["data"])
    data = _merge_clip_data(response["data"], supplemental_data) if isinstance(response.get("data"), dict) else response["data"]
    response = {**response, "data": data, "supplemental_responses": supplemental_responses}
    narration_sentences, narration_audit = load_narration_sentences(cfg.get("processed_root"), _clip_id(row))
    intervals_path = Path(cfg.get("processed_root", "data/processed")) / _clip_id(row) / "intervals.json"
    intervals = read_json(intervals_path) if intervals_path.exists() else None
    chart_context = {**context, "chart_metadata": data if isinstance(data, dict) else {}}
    dynamic = build_dynamic_records(
        clip_id=_clip_id(row),
        visual_data=data if isinstance(data, dict) else {},
        frame_context=frame_context,
        image_paths=image_paths,
        narration_sentences=narration_sentences,
        chart_context=chart_context,
        intervals=intervals,
        audit=[
            {
                "stage": "visual_qwen",
                "model_status": response.get("model_status"),
                "failure_reason": response.get("failure_reason"),
                "supplemental_anchor_count": len(supplemental_responses),
                "supplemental_failures": [
                    item.get("failure_reason")
                    for item in supplemental_responses
                    if item.get("model_status") != "qwen" or item.get("failure_reason")
                ],
            },
            {
                "stage": "narration",
                **narration_audit,
            },
        ],
    )
    dynamic_paths = write_dynamic_outputs(out_dir, dynamic)
    semantic_state_inputs = _write_semantic_state_inputs(dynamic, source_video, out_dir, cfg, force=force)
    dynamic_rows = dynamic.get("states", [])
    rows = data.get("rows") if isinstance(data, dict) else []
    manual_rows = data.get("manual_stub_rows") if isinstance(data, dict) else []
    has_extractable = bool(dynamic.get("include_in_dataset"))

    if has_extractable:
        write_csv(csv_path, dynamic_rows)
    elif csv_path.exists() and force:
        csv_path.unlink()
    if manual_rows:
        write_csv(manual_csv_path, manual_rows)
    elif manual_csv_path.exists() and force:
        manual_csv_path.unlink()
    if not has_extractable and manual_rows:
        write_jsonl(events_path, manual_rows)

    metadata = {
        "title": data.get("title"),
        "chart_type": data.get("chart_type", row.get("chart_type")),
        "unit": data.get("unit"),
        "x_axis": data.get("x_axis"),
        "y_axis": data.get("y_axis"),
        "series": data.get("series", []),
        "entities": _metadata_entities(data) if isinstance(data, dict) else [],
        "visible_text": data.get("visible_text", []),
        "needs_manual_data": bool(data.get("needs_manual_data")) or bool(manual_rows),
        "model_status": response["model_status"],
        "failure_reason": response["failure_reason"],
        "skip_reason": dynamic.get("exclude_reason") if not has_extractable else None,
    }
    validation = {
        "valid_schema": isinstance(data, dict) and isinstance(rows, list),
        "has_extractable_data": has_extractable,
        "value_count": dynamic.get("numeric_fact_count", 0) if has_extractable else 0,
        "numeric_fact_count": dynamic.get("numeric_fact_count", 0),
        "data_completeness": dynamic.get("data_completeness"),
        "dynamic_data": dynamic.get("dynamic_data"),
        "data_change_count": dynamic.get("data_change_count", 0),
        "csv_path": str(csv_path) if has_extractable else None,
        "dynamic_data_json": dynamic_paths["dynamic_data_json"],
        "dynamic_data_csv": dynamic_paths["dynamic_data_csv"],
        "final_data_table_csv": dynamic_paths["final_data_table_csv"],
        "data_change_events_csv": dynamic_paths["data_change_events_csv"],
        "semantic_state_inputs": semantic_state_inputs,
        "manual_csv_path": str(manual_csv_path) if manual_rows else None,
        "manual_stub_count": len(manual_rows) if isinstance(manual_rows, list) else 0,
        "source_frame_count": len(image_paths),
        "sampling": sampling_plan,
        "uncertain_fields": sorted(set(data.get("uncertain_fields", []))),
        "exclude_reason": dynamic.get("exclude_reason"),
        "review_statuses": sorted({str(item.get("review_status")) for item in dynamic_rows}),
        "do_not_use_for_training_without_review": bool(data.get("uncertain_fields")) or bool(manual_rows) or bool(dynamic.get("excluded")),
    }
    write_json(
        raw_path,
        {
            "response": response,
            "metadata": metadata,
            "frame_context": frame_context,
            "image_paths": image_paths,
            "dynamic_data": dynamic,
            "semantic_state_inputs": semantic_state_inputs,
            "narration_audit": narration_audit,
            "sampling": sampling_plan,
            "model_path": scorer.model_path,
            "prompt_version": cfg["model"]["prompt_version"],
        },
    )
    write_json(out_dir / "chart_metadata.json", metadata)
    write_json(validation_path, validation)
    return {
        "data": data,
        "dynamic_data": dynamic,
        "metadata": metadata,
        "validation": validation,
        "csv_path": validation["csv_path"],
        "dynamic_data_json": dynamic_paths["dynamic_data_json"],
        "dynamic_data_csv": dynamic_paths["dynamic_data_csv"],
        "final_data_table_csv": dynamic_paths["final_data_table_csv"],
        "data_change_events_csv": dynamic_paths["data_change_events_csv"],
        "semantic_state_inputs": semantic_state_inputs,
        "manual_csv_path": validation["manual_csv_path"],
        "events_path": str(events_path),
    }
