from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

from datavideo.context import create_context_media
from .animation import detect_animation, reconcile_intent_with_data
from datavideo.cv_align import _looks_like_value_label, detect_bar_states, run_cv_align
from datavideo.cv_align import run_cv_align_line
from datavideo.cv_align import reconcile_line_dynamic
from datavideo.cv_align import read_frame_title
from datavideo.cv_align import read_series_label
from datavideo.cv_reconcile import reconcile_dynamic_data
from datavideo.cv_reconcile import write_dynamic_outputs
from datavideo.chart_processors import SUPPORTED_PROCESSORS, detect_chart_type
from datavideo.metadata import read_clip_rows
from datavideo.narration import transcribe_context_audio
from datavideo.semantic_render import (
    frame_title_status,
    match_chart_style,
    metadata_from_dynamic,
    prefer_frame_visible_title,
    render_data_driven,
    render_data_driven_line,
    resolve_render_title,
)
from datavideo.schemas import ensure_dir, read_json, write_json, write_jsonl

from .multichart_assets import (
    _keyframe_timestamp,
    recover_clip_data,
    select_keyframe,
)
from .multichart_qwen import MultichartQwenClient


def _clip_id(row: dict[str, Any]) -> str:
    return str(
        row.get("output_stem")
        or row.get("clip_id")
        or f"{row.get('chart_type') or 'chart'}_{row.get('chart_index') or 0}"
    )


def _series_label_from_title(title: Any) -> str:
    """Derive a short series name from a chart title.

    "Net additions' in England" -> "Net additions"; titles without a location/
    time qualifier are returned unchanged. This is only a fallback after the
    VLM series field and the vision legend reading.
    """
    text = str(title or "").strip().strip("\"'`.,锛屻€?")
    if not text:
        return ""
    for separator in (" in ", " for ", " from ", " over "):
        if separator in text:
            text = text.split(separator, 1)[0]
            break
    return text.strip().strip("'\"`").strip()[:60]


def _reference_clip_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "clip_id": _clip_id(row),
        "source_video": row.get("output_path"),
    }


def _load_rows(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return read_clip_rows(cfg)


def _selected_keyframe_path(keyframes: dict[str, Any]) -> Path | None:
    assets = keyframes.get("assets") if isinstance(keyframes.get("assets"), dict) else {}
    selected = assets.get("selected")
    if selected:
        path = Path(selected)
        if path.exists():
            return path
    states = keyframes.get("states") if isinstance(keyframes.get("states"), list) else []
    for state in states:
        if isinstance(state, dict) and state.get("asset"):
            path = Path(state["asset"])
            if path.exists():
                return path
    return None


def _cv_align_enabled(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    align_cfg = cfg.get("cv_align") if isinstance(cfg.get("cv_align"), dict) else {}
    if align_cfg.get("enabled") is False:
        return False
    chart_type = str(row.get("chart_type") or "").lower()
    # combined multichart clips frequently contain bar charts, so keep CV
    # alignment enabled for them too.
    return "bar" in chart_type or "combined" in chart_type


def _write_candidate_report(
    clip_root: Path,
    row: dict[str, Any],
    media: dict[str, Any],
    intervals: dict[str, Any],
    asr_report: dict[str, Any],
    keyframes: dict[str, Any],
    animation: dict[str, Any],
    semantic: dict[str, Any],
    chart_data: dict[str, Any],
    semantic_state_svgs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clip_payload = _reference_clip_metadata(row)
    clip_payload["animation_description"] = animation.get("overall_description")
    clip_payload["animation_action_count"] = len(animation.get("major_actions", [])) if isinstance(animation.get("major_actions"), list) else 0
    clip_payload["animation_confidence"] = animation.get("confidence")
    clip_payload["is_target_chart_related"] = animation.get("is_target_chart_related")
    clip_report = {
        "clip": clip_payload,
        "context": media,
        "intervals": intervals,
        "asr": asr_report,
        "clip_video": str(clip_root / "clip.mp4"),
        "keyframes": keyframes,
        "animation_detection": animation,
        "semantic": semantic,
        "semantic_state_svgs": semantic_state_svgs or {},
        "chart_data": chart_data,
    }
    write_json(clip_root / "clip_report.json", clip_report)
    return clip_report


def run_context_pipeline(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    rows = _load_rows(cfg)
    processed_root = ensure_dir(cfg.get("processed_root", "data/processed"))
    results = []
    failures = []
    for row in rows:
        clip_id = _clip_id(row)
        try:
            media = create_context_media({**cfg, "processed_root": str(processed_root)}, row, force=force)
            results.append({"clip_id": clip_id, **media})
        except Exception as exc:
            failure = {"clip_id": clip_id, "failure_reason": str(exc)}
            failures.append(failure)
            write_json(processed_root / clip_id / "context_failed.json", failure)
    report = {"clip_count": len(rows), "completed_count": len(results), "failure_count": len(failures), "clips": results, "failures": failures}
    write_json(processed_root / "multichart_v2_context_report.json", report)
    return report


def run_asr_pipeline(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    rows = _load_rows(cfg)
    processed_root = ensure_dir(cfg.get("processed_root", "data/processed"))
    results = []
    failures = []
    for row in rows:
        clip_id = _clip_id(row)
        processed_dir = ensure_dir(processed_root / clip_id)
        try:
            if not (processed_dir / "intervals.json").exists() or not (processed_dir / "context_audio_16k_mono.wav").exists():
                create_context_media({**cfg, "processed_root": str(processed_root)}, row, force=force)
            intervals = read_json(processed_dir / "intervals.json")
            report = transcribe_context_audio(
                cfg,
                clip_id,
                processed_dir / "context_audio_16k_mono.wav",
                intervals,
                processed_dir,
                force=force,
            )
            results.append(report)
        except Exception as exc:
            failure = {"clip_id": clip_id, "failure_reason": str(exc)}
            failures.append(failure)
            write_json(processed_dir / "narration" / "asr_failed.json", failure)
    report = {"clip_count": len(rows), "completed_count": len(results), "failure_count": len(failures), "clips": results, "failures": failures}
    write_json(processed_root / "multichart_v2_asr_report.json", report)
    return report


def _as_number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _line_metadata_from_dynamic(
    dynamic: dict[str, Any],
    *,
    title: str,
    unit: str = "",
    x_labels: list[str] | None = None,
) -> dict[str, Any]:
    series_values: dict[str, list[float]] = {}
    series_x_labels: dict[str, list[str]] = {}
    inferred_unit = unit
    for row_item in dynamic.get("states") or []:
        if not isinstance(row_item, dict):
            continue
        value = _as_number(row_item.get("value"))
        if value is None:
            continue
        if not inferred_unit:
            inferred_unit = str(row_item.get("unit") or "")
        entity = str(row_item.get("entity") or row_item.get("entity_id") or "series")
        series_values.setdefault(entity, []).append(value)
        series_x_labels.setdefault(entity, []).append(str(row_item.get("state_key") or row_item.get("x_label") or ""))
    inferred_x_labels = x_labels or []
    if not inferred_x_labels:
        seen: set[str] = set()
        for labels in series_x_labels.values():
            for label in labels:
                if label and label not in seen:
                    seen.add(label)
                    inferred_x_labels.append(label)
    return {
        "title": title,
        "unit": inferred_unit,
        "chart_type": "line",
        "x_labels": inferred_x_labels,
        "series": [
            {"name": name, "values": values, "x_labels": series_x_labels.get(name, [])}
            for name, values in series_values.items()
        ],
    }


def _safe_state_key(key: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "-", key.lower()).strip("-")
    return safe or "state"


def _numeric_year(value: Any) -> float | None:
    match = re.search(r"(?:19|20)\d{2}", str(value or ""))
    return float(match.group(0)) if match else None


def _state_groups(dynamic: dict[str, Any] | None) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Return ordered ``(state_key, state_label, rows)`` groups from dynamic data."""
    states = dynamic.get("states") if isinstance(dynamic, dict) else []
    if not isinstance(states, list):
        return []
    chart_type = str(dynamic.get("chart_type") or "").lower() if isinstance(dynamic, dict) else ""
    if chart_type in {"line", "area", "scatter"}:
        visual_ranges = set()
        for row in states:
            if not isinstance(row, dict):
                continue
            if str(row.get("source_type") or "") not in {"visual", "visual_frame_align"}:
                continue
            visual_ranges.add((_as_number(row.get("state_start")), _as_number(row.get("state_end"))))
        if len(visual_ranges) <= 1:
            return []
    # Only an explicit printed state label (year/period read from the frame)
    # defines a state.  Rows without one (auto state_id from animation frames
    # or duplicates of a static chart) never become state groups.
    groups: dict[str, list[dict[str, Any]]] = {}
    order: dict[str, float] = {}
    labels: dict[str, str] = {}
    for row in states:
        if not isinstance(row, dict):
            continue
        key = str(row.get("state_key") or row.get("state_label") or "")
        if not key:
            continue
        start = _as_number(row.get("state_start"))
        groups.setdefault(key, []).append(row)
        order[key] = min(order.get(key, start if start is not None else 0.0), start if start is not None else 0.0)
        labels[key] = str(row.get("state_label") or row.get("state_key") or labels.get(key, key))
    if not groups:
        return []
    # Real states re-render the same entities under different labels (1990 vs
    # 2017); groups with disjoint entities (e.g. years used as categories) are
    # one static chart, not multiple states.
    group_ids = [
        {str(row.get("entity_id") or "") for row in rows if row.get("entity_id")}
        for rows in groups.values()
    ]
    shared = any(
        group_ids[i] & group_ids[j]
        for i in range(len(group_ids))
        for j in range(i + 1, len(group_ids))
    )
    if len(groups) >= 2 and not shared:
        return []
    return [
        (key, labels.get(key, key), groups[key])
        for key in sorted(
            groups,
            key=lambda item: (
                _numeric_year(item) is None,
                _numeric_year(item) or 0.0,
                order[item],
                item,
            ),
        )
    ]


def _find_state_render_dir(
    clip_root: Path,
    state_key: str,
    rows: list[dict[str, Any]],
) -> Path | None:
    """Locate the data-driven semantic output dir for one state."""
    safe = _safe_state_key(state_key)
    candidates = [clip_root / "semantic_states" / safe]
    for row in rows:
        state_id = str(row.get("state_id") or "")
        if not state_id:
            continue
        label = str(row.get("state_label") or state_key)
        safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_") or state_id
        candidates.append(clip_root / "semantic_states" / f"{state_id}_{safe_label}")
        candidates.append(clip_root / "semantic_states" / f"{state_id}_{safe}")
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "semantic.svg").is_file():
            return candidate
    return None


def _pick_primary_state(
    clip_report: dict[str, Any],
    groups: list[tuple[str, str, list[dict[str, Any]]]],
) -> str | None:
    """Pick the state that best matches the selected keyframe (CV-verified preferred)."""
    if not groups:
        return None
    keyframes = clip_report.get("keyframes") or {}
    selected_ts = _as_number((keyframes.get("timestamps") or {}).get("selected"))
    if selected_ts is not None:
        best_key: str | None = None
        best_delta: float | None = None
        for key, _, rows in groups:
            for row in rows:
                start = _as_number(row.get("state_start"))
                if start is None:
                    continue
                delta = abs(start - selected_ts)
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_key = key
        if best_key is not None and best_delta is not None and best_delta <= 1.0:
            return best_key
    for key, _, rows in groups:
        if any(str(row.get("source_type") or "") == "visual_frame_align" for row in rows):
            return key
    return groups[-1][0]


def _metadata_from_cv_report(
    cv_report: dict[str, Any],
    base_metadata: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build render metadata for one plateau state from its CV align report."""
    bars = cv_report.get("bars") or []
    series = []
    for bar in bars:
        label = str(bar.get("label") or bar.get("entity_id") or "").strip()
        value = bar.get("value")
        if not label or not isinstance(value, (int, float)):
            continue
        series.append(
            {
                "name": label,
                "entity_id": bar.get("entity_id"),
                "values": [float(value)],
                "metric": "",
            }
        )
    if not series:
        return None
    standard = cv_report.get("standard_table") or {}
    base = base_metadata or {}
    return {
        "title": str(standard.get("title") or base.get("title") or "Data Chart"),
        "chart_type": "bar",
        "unit": str(standard.get("unit") or base.get("unit") or ""),
        "series": series,
        "orientation": str(cv_report.get("orientation") or "vertical"),
    }


def _write_state_csv(path: Path, cv_report: dict[str, Any], clip_id: str) -> bool:
    fieldnames = [
        "clip_id",
        "entity_id",
        "entity",
        "metric",
        "value",
        "unit",
        "type",
        "source_type",
        "confidence",
        "review_status",
    ]
    rows = []
    for bar in cv_report.get("bars") or []:
        if bar.get("value") is None:
            continue
        rows.append(
            {
                "clip_id": clip_id,
                "entity_id": bar.get("entity_id"),
                "entity": bar.get("label"),
                "metric": "value",
                "value": bar.get("value"),
                "unit": bar.get("unit")
                or (cv_report.get("standard_table") or {}).get("unit")
                or "",
                "type": bar.get("value_type")
                or ("exact" if bar.get("value_read_verified") else "estimated"),
                "source_type": "visual_frame_align",
                "confidence": 0.85 if bar.get("value_read_verified") else 0.6,
                "review_status": "machine",
            }
        )
    if not rows:
        return False
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return True


def _write_state_intent(path: Path, clip_id: str, state_id: str, state: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    write_json(
        path,
        {
            "clip_id": clip_id,
            "state_key": state_id,
            "state_label": state_id,
            "chart_type": "bar",
            "is_static": True,
            "static_description": (
                f"柱状图稳定状态 {state_id}（几何检测：t="
                f"{state.get('start')}-{state.get('end')}s）。"
            ),
            "source": "plateau_state_detection",
        },
    )


def _build_plateau_state_renders(
    clip_id: str,
    candidate_clip: Path,
    cfg: dict[str, Any],
    cv_report: dict[str, Any],
    entities: list[dict[str, Any]],
    chart_data: dict[str, Any],
    clip_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    """Run geometry-based plateau state detection and, when multiple real
    states exist, generate one independent data-driven SVG per state.

    Runs for every bar clip regardless of whether the recovered table was
    reconciled (a clip with only estimated values still needs state
    detection).  The VLM's state/year labels are never used.
    """
    plateau_states: list[dict[str, Any]] = []
    error: str | None = None
    try:
        plateau_states = detect_bar_states(
            candidate_clip,
            cfg,
            expected_bar_count=(cv_report or {}).get("detected_bar_count") or 0,
            out_dir=clip_root,
        )
    except Exception as exc:
        error = str(exc)
    state_renders: list[dict[str, Any]] = []
    if len(plateau_states) >= 2:
        base_metadata = chart_data.get("metadata") or {}
        for state in plateau_states:
            state_id = str(state.get("state_id") or "state")
            rep = Path(str(state.get("representative_path") or ""))
            if not rep.exists():
                continue
            try:
                state_report = run_cv_align(
                    clip_id,
                    rep,
                    entities,
                    ensure_dir(clip_root / "state_align" / state_id),
                    client=None,
                    cfg=cfg,
                )
            except Exception as exc:
                state_renders.append(
                    {
                        "state_key": state_id,
                        "state_dir": str(clip_root / "semantic_states" / state_id),
                        "success": False,
                        "failure_reason": f"{state_id}: {exc}",
                    }
                )
                continue
            state_dir = ensure_dir(clip_root / "semantic_states" / state_id)
            state_md = _metadata_from_cv_report(state_report, base_metadata)
            if state_md is not None:
                render = render_data_driven(clip_id, state_md, state_dir)
            else:
                render = {"success": False, "failure_reason": "no_recoverable_entities"}
            _write_state_csv(state_dir / "data_table.csv", state_report, clip_id)
            _write_state_intent(state_dir / "intent.json", clip_id, state_id, state)
            state_renders.append(
                {
                    "state_key": state_id,
                    "state_dir": str(state_dir),
                    "start": state.get("start"),
                    "end": state.get("end"),
                    **render,
                }
            )
    return plateau_states, state_renders, error


def _cv_geometry(cv_report: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Convert a CV align report's bars into renderer geometry boxes."""
    bars = (cv_report or {}).get("bars") or []
    out = []
    for b in bars:
        if b.get("x") is None or b.get("w") is None:
            continue
        out.append(
            {
                "entity_id": b.get("entity_id"),
                "label": b.get("label"),
                "x": b.get("x"),
                "y": b.get("y"),
                "w": b.get("w"),
                "h": b.get("h"),
            }
        )
    return out


def _copy_if_exists(src: Path, dst: Path, written: dict[str, str], dst_name: str) -> None:
    if src.exists() and src.stat().st_size > 0:
        shutil.copy2(src, dst)
        written[dst_name] = str(dst)


def _write_state_table(
    clip_root: Path,
    rows: list[dict[str, Any]],
    out_path: Path,
) -> bool:
    csv_path = clip_root / "dynamic_data.csv"
    if csv_path.exists():
        wanted = {
            str(row.get("state_key") or row.get("state_label") or row.get("state_id"))
            for row in rows
        }
        state_ids = {str(row.get("state_id")) for row in rows if row.get("state_id")}
        kept = []
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for record in csv.DictReader(f):
                if str(record.get("state_key") or record.get("state_label") or record.get("state_id")) in wanted:
                    kept.append(record)
                elif state_ids and str(record.get("state_id")) in state_ids:
                    kept.append(record)
        if kept:
            with out_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
                writer.writeheader()
                writer.writerows(kept)
            return True
    if not rows:
        return False
    fields = [
        "clip_id",
        "state_id",
        "state_key",
        "state_label",
        "entity",
        "entity_id",
        "metric",
        "value",
        "unit",
        "value_type",
        "source_type",
        "confidence",
        "review_status",
        "state_start",
        "state_end",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return True


def build_dataset_folder(
    clip_root: str | Path,
    clip_report: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a self-contained ``dataset/`` folder for one clip.

    For dynamic clips (multiple recovered data states) the folder contains a
    top-level full dynamic data table + reconciled animation intent, and one
    ``states/<state_key>/`` subfolder per state with that state's semantic.svg,
    components svg, keyframe, data rows and a static intent. Static clips keep
    a single flat sample (semantic.svg + data_table.csv + intent.json).
    """
    clip_root = Path(clip_root)
    out = clip_root / "dataset"
    if out.exists():
        shutil.rmtree(out)
    out = ensure_dir(out)
    written: dict[str, str] = {}

    dynamic = None
    dynamic_path = clip_root / "dynamic_data.json"
    if dynamic_path.exists():
        try:
            dynamic = json.loads(dynamic_path.read_text(encoding="utf-8"))
        except Exception:
            dynamic = None
    # Geometry-based plateau states (bar clips) are authoritative when the
    # state scan ran; the VLM's state/year labels are never used for them.
    plateau_states: list[dict[str, Any]] = []
    plateau_path = clip_root / "state_scan_report.json"
    plateau_mode = plateau_path.exists()
    if plateau_mode:
        try:
            plateau_states = (json.loads(plateau_path.read_text(encoding="utf-8")) or {}).get("states") or []
        except Exception:
            plateau_states = []
    if plateau_mode:
        groups = [(str(state.get("state_id") or "state"), str(state.get("state_id") or "state"), []) for state in plateau_states]
    else:
        groups = [
            (key, label, rows)
            for key, label, rows in _state_groups(dynamic)
            if any(
                str(row.get("source_type") or "") in {"visual", "visual_frame_align"}
                for row in rows
            )
        ]
    dynamic_packaging = len(groups) >= 2

    intent = None
    intent_path = clip_root / "animation_detection.json"
    if intent_path.exists():
        try:
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
        except Exception:
            intent = None
    if intent is not None and dynamic is not None:
        intent = reconcile_intent_with_data(intent, dynamic)
    if intent is not None:
        write_json(out / "intent.json", intent)
        written["intent.json"] = str(out / "intent.json")

    if dynamic_packaging:
        table_src = clip_root / "dynamic_data.csv"
        if table_src.exists() and table_src.stat().st_size > 0:
            shutil.copy2(table_src, out / "data_table.csv")
            written["data_table.csv"] = str(out / "data_table.csv")
    if "data_table.csv" not in written:
        _copy_if_exists(clip_root / "final_data_table.csv", out / "data_table.csv", written, "data_table.csv")

    nar = None
    processed_dir = (clip_report.get("context") or {}).get("processed_dir")
    if processed_dir:
        nar = Path(processed_dir) / "narration" / "selected_full_sentences.jsonl"
    if nar is None:
        asr_path = (clip_report.get("asr") or {}).get("path")
        if asr_path:
            nar = Path(asr_path).parent / "selected_full_sentences.jsonl"
    if nar is not None and nar.exists() and nar.stat().st_size > 0:
        shutil.copy2(nar, out / "narration.jsonl")
        written["narration.jsonl"] = str(out / "narration.jsonl")

    keyframes = clip_report.get("keyframes") or {}
    assets = keyframes.get("assets") if isinstance(keyframes.get("assets"), dict) else {}
    selected = assets.get("selected")
    if selected and Path(selected).exists():
        shutil.copy2(Path(selected), out / "keyframe.png")
        written["keyframe.png"] = str(out / "keyframe.png")
    _copy_if_exists(clip_root / "aligned_overlay.png", out / "aligned_overlay.png", written, "aligned_overlay.png")

    primary_dir = None
    if dynamic_packaging:
        primary_key = _pick_primary_state(clip_report, groups)
        if primary_key is not None:
            primary_rows = next((rows for key, _, rows in groups if key == primary_key), [])
            primary_dir = _find_state_render_dir(clip_root, primary_key, primary_rows)
    if primary_dir is not None:
        _copy_if_exists(primary_dir / "semantic.svg", out / "semantic.svg", written, "semantic.svg")
        _copy_if_exists(primary_dir / "semantic_components.svg", out / "semantic_components.svg", written, "semantic_components.svg")
    if "semantic.svg" not in written:
        _copy_if_exists(clip_root / "semantic.svg", out / "semantic.svg", written, "semantic.svg")
    if "semantic_components.svg" not in written:
        _copy_if_exists(clip_root / "semantic_components.svg", out / "semantic_components.svg", written, "semantic_components.svg")

    clip = clip_report.get("clip") or {}
    chart_type = str(clip.get("chart_type") or "")
    clip_id = str(clip.get("clip_id") or "")
    state_entries = []
    if dynamic_packaging:
        for state_key, state_label, rows in groups:
            safe = _safe_state_key(state_key)
            state_out = ensure_dir(out / "states" / safe)
            state_files: dict[str, str] = {}
            # Each state is an independent data-driven SVG generation; the
            # per-state folder only packages the SVG and its data table
            # (manual adjustment is expected).  No per-state keyframe or
            # VLM semantic-component files are produced anymore.
            if plateau_mode:
                render_dir = clip_root / "semantic_states" / state_key
            else:
                render_dir = _find_state_render_dir(clip_root, state_key, rows)
            if render_dir is not None:
                _copy_if_exists(render_dir / "semantic.svg", state_out / "semantic.svg", state_files, "semantic.svg")
            if plateau_mode:
                # Plateau states carry their own data table + intent written
                # by the pipeline (vision-verified per-state values).
                _copy_if_exists(render_dir / "data_table.csv", state_out / "data_table.csv", state_files, "data_table.csv")
                _copy_if_exists(render_dir / "intent.json", state_out / "intent.json", state_files, "intent.json")
            else:
                if _write_state_table(clip_root, rows, state_out / "data_table.csv"):
                    state_files["data_table.csv"] = str(state_out / "data_table.csv")
                metric = str(rows[0].get("metric") or "指标") if rows else "指标"
                static_intent = {
                    "clip_id": clip_id,
                    "state_key": state_key,
                    "state_label": state_label,
                    "chart_type": chart_type,
                    "is_static": True,
                    "static_description": f"渲染{state_label}年的{metric}图表（静态状态快照）。",
                    "source": "static_state_snapshot",
                }
                write_json(state_out / "intent.json", static_intent)
                state_files["intent.json"] = str(state_out / "intent.json")
            state_entries.append(
                {
                    "state_key": state_key,
                    "state_label": state_label,
                    "dir": str(state_out),
                    "entity_count": len(rows),
                    "files": state_files,
                }
            )

    values = []
    for _, _, rows in groups:
        for row in rows:
            values.append(
                {
                    "state_key": row.get("state_key") or row.get("state_label"),
                    "entity": row.get("entity"),
                    "value": row.get("value"),
                    "type": row.get("value_type"),
                    "confidence": row.get("confidence"),
                }
            )
    if not values:
        table = clip_root / "final_data_table.csv"
        if table.exists():
            with table.open("r", encoding="utf-8-sig", newline="") as f:
                for record in csv.DictReader(f):
                    values.append(
                        {
                            "state_key": None,
                            "entity": record.get("entity"),
                            "value": record.get("value"),
                            "type": record.get("type"),
                            "confidence": record.get("confidence"),
                        }
                    )

    manifest = {
        "clip_id": clip_id,
        "title": clip.get("raw_video_title"),
        "chart_type": chart_type,
        "source_time_range": {
            "start": clip.get("start_seconds"),
            "end": clip.get("end_seconds"),
        },
        "needs_review": bool(keyframes.get("needs_review")),
        "boundary_reason": keyframes.get("boundary_reason"),
        "animation_description": (intent or {}).get("overall_description"),
        "intent_reconciled_with_data": bool((intent or {}).get("reconciled_with_data")),
        "data_state_count": len(groups),
        "states": state_entries,
        "values": values,
        "files": written,
    }
    write_json(out / "manifest.json", manifest)
    written["manifest.json"] = str(out / "manifest.json")
    return {"dataset_dir": str(out), "files": written, "state_count": len(groups)}


def run_pipeline(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    processed_root = ensure_dir(cfg.get("processed_root", "data/processed"))
    generated_root = ensure_dir(cfg.get("generated_root", "data/generated"))
    rows = _load_rows(cfg)
    client = MultichartQwenClient(cfg)

    clip_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in rows:
        clip_id = _clip_id(row)
        processed_dir = ensure_dir(processed_root / clip_id)
        clip_root = ensure_dir(generated_root / clip_id)

        try:
            media = create_context_media({**cfg, "processed_root": str(processed_root)}, row, force=force)
            intervals = media["intervals"]
            visual_clip = Path(media["visual_clip"])
            if not visual_clip.exists():
                raise RuntimeError("missing_visual_clip: run `context` first")
            media = {
                **media,
                "visual_clip": str(visual_clip),
                "intervals": intervals,
            }
            asr_report_path = processed_dir / "narration" / "asr_report.json"
            asr_report = read_json(asr_report_path) if asr_report_path.exists() else {"status": "missing", "path": str(asr_report_path)}
            prior_report_path = clip_root / "clip_report.json"
            prior_visual = read_json(prior_report_path).get("clip", {}).get("visual_clip_path") if prior_report_path.exists() else None
            asset_force = force or prior_visual != str(visual_clip)
            candidate_clip = clip_root / "clip.mp4"
            if asset_force or not candidate_clip.exists():
                import shutil

                shutil.copy2(visual_clip, candidate_clip)

            keyframes = select_keyframe(
                candidate_clip,
                _reference_clip_metadata(row),
                clip_root / "keyframes",
                {**cfg, "processed_root": str(processed_root)},
                client=client,
                force=asset_force,
                context_video=media.get("context_video"),
                context_visual_end=(media.get("intervals") or {}).get("visual_clip_context", {}).get("end"),
            )
            animation = detect_animation(
                {**cfg, "processed_root": str(processed_root)},
                _reference_clip_metadata(row),
                keyframes,
                clip_root,
                client=client,
                force=asset_force,
            )
            chart_data = recover_clip_data(
                cfg,
                keyframes,
                _reference_clip_metadata(row),
                clip_root,
                client=client,
                force=asset_force,
            )
            recovered_type = (chart_data.get("metadata") or {}).get("chart_type") or row.get("chart_type")
            processor, declared_type, type_consistent = detect_chart_type(
                row.get("chart_type"),
                recovered_type,
            )
            semantic = render_data_driven(_clip_id(row), chart_data.get("metadata") or {}, clip_root)
            dynamic = chart_data.get("dynamic_data") or {}
            recovered_type = (chart_data.get("metadata") or {}).get("chart_type") or row.get("chart_type")
            processor, declared_type, type_consistent = detect_chart_type(row.get("chart_type"), recovered_type)
            render_metadata = metadata_from_dynamic(
                dynamic,
                visible_text=(chart_data.get("metadata") or {}).get("visible_text"),
            )
            if processor == "line":
                selected_keyframe = _selected_keyframe_path(keyframes)
                line_report = run_cv_align_line(
                    _clip_id(row),
                    selected_keyframe,
                    clip_root,
                    cfg=cfg,
                )
                original_title = (chart_data.get("metadata") or {}).get("title")
                visible_text = (chart_data.get("metadata") or {}).get("visible_text") or []
                resolved_title = prefer_frame_visible_title(
                    resolve_render_title(original_title, ""),
                    visible_text,
                )
                # The VLM's visible_text can misread the in-frame title (e.g.
                # line_4 read "Lack of building" instead of "'Net additions'
                # in England"). The vision model reads the printed title
                # directly, so for line charts it is the authoritative source.
                if selected_keyframe is not None:
                    try:
                        frame_title = read_frame_title(selected_keyframe, cfg)
                        if frame_title and len(frame_title) >= 3:
                            resolved_title = frame_title
                    except Exception:
                        pass
                series_label = None
                qwen_series = (chart_data.get("metadata") or {}).get("series")
                if isinstance(qwen_series, list) and qwen_series and isinstance(qwen_series[0], dict):
                    candidate = str(qwen_series[0].get("name") or "").strip()
                    if candidate and candidate.lower() not in {"series", "value", "metric", "unknown"}:
                        series_label = candidate
                if not series_label and selected_keyframe is not None:
                    try:
                        series_label = read_series_label(selected_keyframe, cfg)
                    except Exception:
                        series_label = None
                if not series_label:
                    series_label = _series_label_from_title(resolved_title)
                cv_has_values = int(line_report.get("point_count") or 0) > 0
                if cv_has_values:
                    dynamic = reconcile_line_dynamic(
                        line_report.get("lines") or [],
                        clip_id=_clip_id(row),
                        image_path=selected_keyframe,
                        keyframe_timestamp=_keyframe_timestamp(keyframes),
                        unit=line_report.get("tick_unit") or "",
                        series_label=series_label,
                    )
                    chart_data = {**chart_data, "dynamic_data": dynamic}
                line_metadata = _line_metadata_from_dynamic(
                    dynamic,
                    title=resolved_title,
                    unit=line_report.get("tick_unit") or "",
                    x_labels=line_report.get("x_axis_labels") or None,
                )
                semantic = render_data_driven_line(_clip_id(row), line_metadata, clip_root)
                semantic["cv_align"] = line_report
                semantic["reconciled"] = {
                    "line_count": line_report.get("line_count", 0),
                    "point_count": line_report.get("point_count", 0),
                    "used_cv_values": cv_has_values,
                    "fallback_source": None if cv_has_values else "qwen_dynamic_data",
                }
                write_dynamic_outputs(clip_root, dynamic)
                write_json(clip_root / "chart_metadata.json", line_metadata)
            elif _cv_align_enabled(row, cfg) and processor == "bar":
                selected_keyframe = _selected_keyframe_path(keyframes)
                entities: list[dict[str, Any]] = []
                seen: set[str] = set()
                for state_row in (dynamic.get("states") or []) if isinstance(dynamic, dict) else []:
                    if not isinstance(state_row, dict):
                        continue
                    eid = str(state_row.get("entity_id") or "")
                    label = str(state_row.get("entity") or eid)
                    if eid in ("", "unknown") or eid in seen:
                        continue
                    if _looks_like_value_label(label):
                        # A recovered entity that is really an axis tick /
                        # printed value (e.g. "$3,000") is a hallucination and
                        # must never become a bar label.
                        continue
                    seen.add(eid)
                    entities.append({"id": eid, "label": label})
                # Bar/combined clips may have an empty recovered table when
                # the VLM failed (OOM / broken JSON) even though the frame
                # clearly shows a chart with axis tick marks. Run CV alignment
                # anyway: match_entities creates entities from the frame
                # labels, and tick estimation can then recover the values.
                chart_kind = str(row.get("chart_type") or "")
                if selected_keyframe is not None and (entities or "bar" in chart_kind or "combined" in chart_kind):
                    cv_report = run_cv_align(
                        _clip_id(row),
                        selected_keyframe,
                        entities,
                        clip_root,
                        client=client,
                        cfg=cfg,
                    )
                    semantic["cv_align"] = cv_report
                    # Plan B rendering: CV-detected bar boxes give the real
                    # geometry; the vision model supplies the visual style
                    # spec; values still come from the data table.
                    chart_style: dict[str, Any] = {}
                    if selected_keyframe is not None:
                        try:
                            chart_style = match_chart_style(selected_keyframe, cfg)
                        except Exception:
                            chart_style = {}
                    cv_geometry = _cv_geometry(cv_report)
                    if cv_geometry:
                        semantic = render_data_driven(
                            _clip_id(row),
                            chart_data.get("metadata") or {},
                            clip_root,
                            geometry=cv_geometry,
                            style=chart_style,
                        )
                    implausible = cv_report.get("implausible_bars") or []
                    reconciled = None
                    if not implausible:
                        reconciled = reconcile_dynamic_data(
                            dynamic,
                            cv_report,
                            clip_id=_clip_id(row),
                            keyframe_timestamp=_keyframe_timestamp(keyframes),
                            image_path=selected_keyframe,
                            out_dir=clip_root,
                        )
                    else:
                        semantic["reconciled"] = {
                            "updated_bar_count": 0,
                            "skipped_bar_count": len(implausible),
                            "skipped_bars": implausible,
                            "reason": "frame values failed plausibility; kept recovered data table",
                        }
                    if reconciled:
                        dynamic = reconciled["dynamic"]
                        chart_data = {**chart_data, "dynamic_data": dynamic}
                        corrected_metadata = metadata_from_dynamic(
                            dynamic,
                            visible_text=(chart_data.get("metadata") or {}).get("visible_text"),
                        )
                        if corrected_metadata:
                            original_title = (chart_data.get("metadata") or {}).get("title")
                            if original_title:
                                visible_text = (chart_data.get("metadata") or {}).get("visible_text") or []
                                resolved = resolve_render_title(
                                    original_title,
                                    corrected_metadata.get("title"),
                                )
                                final_title = prefer_frame_visible_title(resolved, visible_text)
                                # No usable title candidate in the recovered
                                # visible text (e.g. bar_29, where the VLM
                                # dropped the title line entirely): ask the
                                # vision model to read the frame title.
                                if (
                                    frame_title_status(final_title, visible_text) == "none"
                                ):
                                    # The standard-table read already has the
                                    # in-frame title; only fall back to a
                                    # dedicated title read when it is empty.
                                    frame_title = (cv_report.get("standard_table") or {}).get("title") or ""
                                    if not frame_title and selected_keyframe is not None:
                                        try:
                                            frame_title = read_frame_title(selected_keyframe, cfg)
                                        except Exception:
                                            frame_title = ""
                                    if frame_title:
                                        final_title = frame_title
                                corrected_metadata["title"] = final_title
                            cv_orientation = (cv_report or {}).get("orientation")
                            if cv_orientation:
                                corrected_metadata["orientation"] = cv_orientation
                            write_json(clip_root / "chart_metadata.json", corrected_metadata)
                            chart_data = {**chart_data, "metadata": corrected_metadata}
                            semantic = render_data_driven(
                                _clip_id(row),
                                corrected_metadata,
                                clip_root,
                                geometry=cv_geometry,
                                style=chart_style,
                            )
                        semantic["cv_align"] = cv_report
                        semantic["reconciled"] = {
                            "updated_bar_count": reconciled["updated_bar_count"],
                            "skipped_bar_count": reconciled["skipped_bar_count"],
                            "state_key": reconciled["state_key"],
                            "state_id": reconciled["state_id"],
                        }
                    # Geometry-based state detection runs for every bar clip
                    # (independent of whether the recovered table merged).
                    plateau_states, state_renders, state_error = _build_plateau_state_renders(
                        _clip_id(row),
                        candidate_clip,
                        cfg,
                        cv_report,
                        entities,
                        chart_data,
                        clip_root,
                    )
                    semantic["plateau_states"] = plateau_states
                    semantic["state_renders"] = state_renders
                    if state_error:
                        semantic["state_detection_error"] = state_error
            animation = reconcile_intent_with_data(animation, dynamic)
            write_json(clip_root / "animation_detection.json", animation)
            # Multi-state "changes" are emitted as one data-driven SVG per
            # state (semantic["state_renders"]) instead of running a second
            # VLM semantic-component pass per state.  Per-state outputs are
            # small and manually adjustable, so the heavy automated pass is
            # intentionally dropped.
            semantic_state_svgs: dict[str, Any] = {}

            clip_report = _write_candidate_report(
                clip_root,
                row,
                media,
                intervals,
                asr_report,
                keyframes,
                animation,
                semantic,
                semantic_state_svgs,
                chart_data,
            )
            clip_report["chart_processor"] = processor
            clip_report["chart_type_consistent"] = type_consistent
            clip_report["unsupported_processor"] = processor not in SUPPORTED_PROCESSORS
            clip_report["visual_boundary_source"] = "web_reference_interval"
            clip_report["deprecated_clip_boundary_review_ignored"] = True
            clip_report["asset_status"] = "fresh"
            clip_report["clip"]["visual_clip_path"] = str(visual_clip)
            clip_report["clip"]["visual_clip_source"] = "reference_source"
            write_json(clip_root / "clip_report.json", clip_report)
            clip_report["dataset"] = build_dataset_folder(clip_root, clip_report)
            write_json(clip_root / "clip_report.json", clip_report)
            clip_reports.append(clip_report)
            failed_path = clip_root / "clip_report_failed.json"
            if failed_path.exists():
                failed_path.unlink()
        except Exception as exc:
            failure = {"clip_id": clip_id, "clip": row, "failure_reason": str(exc)}
            write_json(clip_root / "clip_report_failed.json", failure)
            failures.append(failure)

    write_jsonl(generated_root / "multichart_v2_clips.jsonl", [report["clip"] for report in clip_reports])
    run_report = {
        "sample_id": cfg["sample_id"],
        "source": str(cfg.get("clip_metadata_csv") or cfg.get("raw_clips_jsonl", "data/raw/datavideo_clips.jsonl")),
        "clip_count": len(rows),
        "completed_clip_count": len(clip_reports),
        "failure_count": len(failures),
        "processed_root": str(processed_root),
        "generated_root": str(generated_root),
        "clips": clip_reports,
        "failures": failures,
        "config_hash": cfg.get("config_hash"),
    }
    write_json(generated_root / "multichart_v2_run_report.json", run_report)
    return run_report
