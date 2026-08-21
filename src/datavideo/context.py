from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .media import extract_clip_accurate, ffprobe, standardize_media
from .schemas import ensure_dir, read_json, write_json


def _seconds(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def _reference_interval(row: dict[str, Any]) -> dict[str, float]:
    return {
        "start": _seconds(row.get("start_seconds", row.get("start_time"))),
        "end": _seconds(row.get("end_seconds", row.get("end_time"))),
    }


def compute_context_source_interval(
    reference_start: float,
    reference_end: float,
    *,
    padding_before_seconds: float = 5.0,
    padding_after_seconds: float = 5.0,
    source_duration_seconds: float | None = None,
) -> dict[str, float]:
    start = max(0.0, float(reference_start) - float(padding_before_seconds))
    end = float(reference_end) + float(padding_after_seconds)
    if source_duration_seconds is not None:
        end = min(float(source_duration_seconds), end)
    return {"start": round(start, 3), "end": round(max(start, end), 3)}


def _context_interval(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, float]:
    ref = _reference_interval(row)
    padding = cfg.get("context", {})
    return compute_context_source_interval(
        ref["start"],
        ref["end"],
        padding_before_seconds=_seconds(padding.get("padding_before_seconds", 5.0)),
        padding_after_seconds=_seconds(padding.get("padding_after_seconds", 5.0)),
    )


def _download_context_source(row: dict[str, Any], cfg: dict[str, Any], out_path: Path, force: bool) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return True

    context_cfg = cfg.get("context", {})
    cookies = Path(context_cfg.get("cookies", "www.youtube.com_cookies.txt"))
    proxy = context_cfg.get("proxy")
    max_height = int(context_cfg.get("max_height", 720))
    interval = _context_interval(row, cfg)
    section = f"*{_hms(interval['start'])}-{_hms(interval['end'])}"
    format_selector = (
        f"bv*[height<={max_height}][ext=mp4]+ba[ext=m4a]/"
        f"bv*[height<={max_height}][ext=mp4]/"
        f"bv*[height<={max_height}]+ba/b[height<={max_height}]"
    )
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--cookies",
        str(cookies),
        "--js-runtimes",
        "node",
        "--remote-components",
        "ejs:github",
        "--download-sections",
        section,
        "--force-keyframes-at-cuts",
        "-f",
        format_selector,
        "--merge-output-format",
        "mp4",
        "-o",
        str(out_path.with_suffix(".%(ext)s")),
        row["youtube_url"],
    ]
    if proxy:
        cmd[3:3] = ["--proxy", str(proxy)]
    if force:
        cmd.insert(-1, "--force-overwrites")
    subprocess.run(cmd, check=True)
    return out_path.exists() and out_path.stat().st_size > 0


def build_intervals_payload(
    row: dict[str, Any],
    cfg: dict[str, Any],
    *,
    context_source: dict[str, float],
    context_duration: float,
    requires_context_redownload: bool,
    boundary_reason: str = "",
    needs_review: bool = False,
) -> dict[str, Any]:
    reference = _reference_interval(row)
    visual_start = max(0.0, reference["start"] - context_source["start"])
    visual_duration = max(0.0, reference["end"] - reference["start"])
    visual_context = {"start": round(visual_start, 3), "end": round(visual_start + visual_duration, 3)}
    return {
        "time_unit": "seconds",
        "reference_source": reference,
        "context_source": context_source,
        "visual_clip_context": visual_context,
        "visual_clip_relative": {"start": 0.0, "end": round(visual_duration, 3)},
        "boundary_reason": boundary_reason,
        "needs_review": bool(needs_review),
        "requires_context_redownload": bool(requires_context_redownload),
        "context_duration_seconds": round(float(context_duration), 3),
        "context_padding_before_seconds": _seconds(cfg.get("context", {}).get("padding_before_seconds", 5.0)),
        "context_padding_after_seconds": _seconds(cfg.get("context", {}).get("padding_after_seconds", 5.0)),
    }


def _canonicalize_cached_intervals(row: dict[str, Any], cfg: dict[str, Any], intervals: dict[str, Any], visual_clip: Path) -> dict[str, Any]:
    context_source = intervals.get("context_source") or _reference_interval(row)
    canonical = build_intervals_payload(
        row,
        cfg,
        context_source=context_source,
        context_duration=float(intervals.get("context_duration_seconds") or 0.0),
        requires_context_redownload=bool(intervals.get("requires_context_redownload")),
        boundary_reason=str(intervals.get("boundary_reason", "") or ""),
        needs_review=bool(intervals.get("needs_review")),
    )
    if visual_clip.exists():
        try:
            probe = ffprobe(visual_clip)
            actual_duration = float(probe["format"]["duration"])
            expected_duration = float(canonical["visual_clip_relative"]["end"])
            canonical["visual_clip"] = {
                "path": str(visual_clip),
                "source": "context.mp4",
                "expected_duration_seconds": round(expected_duration, 3),
                "actual_duration_seconds": round(actual_duration, 3),
                "duration_error_seconds": round(actual_duration - expected_duration, 3),
                "sync_check": "ffprobe_duration_recorded",
            }
        except Exception:
            canonical["visual_clip"] = {"path": str(visual_clip), "source": "context.mp4", "sync_check": "ffprobe_failed"}
    return canonical


def create_context_media(
    cfg: dict[str, Any],
    row: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    processed = ensure_dir(Path(cfg.get("processed_root", "data/processed")) / row["output_stem"])
    context_video = processed / "context.mp4"
    context_audio = processed / "context_audio_16k_mono.wav"
    visual_clip = processed / "visual_clip.mp4"
    intervals_path = processed / "intervals.json"
    if context_video.exists() and context_audio.exists() and visual_clip.exists() and intervals_path.exists() and not force:
        intervals = read_json(intervals_path)
        intervals = _canonicalize_cached_intervals(row, cfg, intervals, visual_clip)
        write_json(intervals_path, intervals)
        return {
            "context_video": str(context_video),
            "context_audio": str(context_audio),
            "visual_clip": str(visual_clip),
            "context_source_path": str(processed / "context_download.mp4"),
            "intervals": intervals,
            "processed_dir": str(processed),
            "requires_context_redownload": bool(intervals.get("requires_context_redownload")),
        }

    download_source = processed / "context_download.mp4"
    context_source = _context_interval(row, cfg)
    requires_context_redownload = False

    try:
        if _download_context_source(row, cfg, download_source, force=force):
            context_source_path = download_source
        else:
            raise RuntimeError("context download returned no file")
    except Exception:
        requires_context_redownload = True
        context_source_path = Path(row["output_path"])

    media = standardize_media(
        context_source_path,
        processed,
        cfg["video_standardization"],
        video_out=context_video,
        wav_out=context_audio,
        report_name="context_standardization_report.json",
        force=force,
    )
    probe = ffprobe(media["video"])
    duration = float(probe["format"]["duration"])
    if requires_context_redownload:
        ref = _reference_interval(row)
        ctx = _context_interval(row, cfg)
        # The fallback raw video may be either an exact reference clip
        # (duration ~ reference) or a padded context download
        # (duration ~ context interval). Detect which one by its duration
        # so the visual clip is cut from the correct offset.
        if abs(duration - (ctx["end"] - ctx["start"])) < 1.0:
            context_source = ctx
        else:
            context_source = ref
    else:
        context_source["end"] = min(context_source["end"], context_source["start"] + duration)

    intervals = build_intervals_payload(
        row,
        cfg,
        context_source=context_source,
        context_duration=duration,
        requires_context_redownload=requires_context_redownload,
        boundary_reason="context download fallback to exact clip" if requires_context_redownload else "",
        needs_review=requires_context_redownload,
    )
    visual = intervals["visual_clip_context"]
    visual_report = extract_clip_accurate(
        media["video"],
        float(visual["start"]),
        float(visual["end"]),
        visual_clip,
        cfg["video_standardization"],
        force=force,
    )
    intervals["visual_clip"] = {
        "path": str(visual_clip),
        "source": "context.mp4",
        "expected_duration_seconds": visual_report["expected_duration_seconds"],
        "actual_duration_seconds": visual_report["actual_duration_seconds"],
        "duration_error_seconds": visual_report["duration_error_seconds"],
        "sync_check": "ffprobe_duration_recorded",
    }
    write_json(intervals_path, intervals)
    write_json(processed / "visual_clip_report.json", visual_report)
    return {
        "context_video": str(context_video),
        "context_audio": str(context_audio),
        "visual_clip": str(visual_clip),
        "context_source_path": str(context_source_path),
        "intervals": intervals,
        "processed_dir": str(processed),
        "requires_context_redownload": requires_context_redownload,
    }


def load_intervals(path: str | Path) -> dict[str, Any]:
    return read_json(path)
