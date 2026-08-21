"""CV-based bar detection + entity matching + vision value reading.

Produces, from a keyframe and the recovered data table:
  - aligned_overlay.png : the keyframe with boxes drawn on the real bars
  - semantic_aligned.svg: bars placed at the real (pixel) bar coordinates
  - aligned_report.json : boxes, matched entities, value-ratio consistency,
                          vision-based entity order and alignment verification

Values and label order are read by the external vision model (vision.js),
which is far more reliable than whole-frame OCR on the local 3B model.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .frames import extract_frames
from .keyframes import extract_still
from .media import ffprobe
from .schemas import ensure_dir, write_json


# A number token, optionally with thousands separators and a % suffix
# ("30,000", "36.1%", "890").  Shared by the vision crop fallback so values
# such as 43,000 are kept whole instead of being truncated to 43.
_NUMBER_TOKEN_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?\s*%?")


VISION_DEFAULTS = {
    "node_path": os.environ.get("DATAVIDEO_VISION_NODE", "node"),
    "script": os.environ.get("DATAVIDEO_VISION_SCRIPT", "vision.js"),
    "proxy": None,
}


def _call_vision(
    image_path: str | Path,
    prompt: str,
    cfg: dict[str, Any] | None = None,
    *,
    temperature: float | None = None,
) -> str:
    """Call the external vision model (vision.js) on one image and return text.

    Tries a direct connection first (DashScope is reachable without a proxy in
    most environments); if a proxy is configured in ``cfg["cv_align"]["proxy"]``
    it is retried through that proxy when the direct attempt fails.
    """
    cfg = cfg or {}
    v = {**VISION_DEFAULTS, **(cfg.get("cv_align") or {})}
    attempts = [None]
    if v.get("proxy"):
        attempts.append(v["proxy"])
    last_error = "vision call failed"
    for proxy in attempts:
        node_parent = str(Path(v["node_path"]).parent)
        path_prefix = "" if node_parent == "." else node_parent + os.pathsep
        env = {
            **os.environ,
            "PATH": path_prefix + os.environ.get("PATH", ""),
        }
        if proxy:
            env["HTTPS_PROXY"] = proxy
            env["HTTP_PROXY"] = proxy
        else:
            env.pop("HTTPS_PROXY", None)
            env.pop("HTTP_PROXY", None)
        try:
            cmd = [v["node_path"], v["script"], str(image_path), prompt]
            if temperature is not None:
                cmd.extend(["--temp", str(temperature)])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=120,
                env=env,
            )
            text = (result.stdout or "").strip() or (result.stderr or "").strip()
            if result.returncode == 0 and text:
                return text
            last_error = text or f"vision.js exit code {result.returncode}"
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(last_error)


def _hue_near(hue: np.ndarray, center: int, tolerance: int) -> np.ndarray:
    """Boolean mask for pixels whose circular hue distance to ``center`` is <= tolerance."""
    # Cast away uint8 first: under NEP 50 (NumPy >= 2) an unsigned subtraction
    # wraps around instead of promoting, which silently truncated the hue band.
    delta = np.abs(hue.astype(np.int32) - center)
    delta = np.minimum(delta, 180 - delta)
    return delta <= tolerance


def _estimate_background_mask(hsv: np.ndarray) -> np.ndarray:
    """Estimate which pixels belong to the chart background.

    The previous heuristic took the dominant hue among *saturated* pixels as
    the background hue and erased everything within +-18 degrees of it. That
    silently deletes the bars whenever the bars themselves are the largest
    saturated region of the frame (e.g. a light-gray chart background with
    solid colored bars, the most common news-graphic layout).

    Instead the background model is built from pixels that are much more
    likely to be background:
      * the outer border strip of the frame (bars almost never touch it), and
      * low-saturation pixels (white/gray/light backgrounds, dark panels).
    When the evidence is achromatic (median saturation < 55) the background is
    a saturation threshold; when it is chromatic (saturated panels, e.g. the
    WeChat test clip) the background is a hue band around the median evidence
    hue, with achromatic pixels always treated as background too.
    """
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    H, W = h.shape
    bw = max(3, W // 40)
    bh = max(3, H // 40)
    border = np.zeros((H, W), dtype=bool)
    border[:bh, :] = True
    border[-bh:, :] = True
    border[:, :bw] = True
    border[:, -bw:] = True

    low_sat = s < 40
    evidence = border | low_sat
    if evidence.sum() == 0:
        # Edge-to-edge saturated frame with no achromatic pixels: fall back to
        # the dominant saturated hue (legacy behaviour).
        sel = (s > 40) & (v > 40)
        if sel.sum() == 0:
            return border
        hist, _ = np.histogram(h[sel], bins=180, range=(0, 180))
        bg_hue = int(np.argmax(hist))
        return _hue_near(h, bg_hue, 18) | border

    if float(np.median(s[evidence])) < 55:
        # Achromatic background (white / light gray / dark panel): the bars are
        # whatever saturated pixels remain.
        sat_tol = max(55, int(np.median(s[evidence])) + 35)
        return (s < sat_tol) | border

    # Chromatic background: circular median hue of the border + achromatic
    # evidence (the panel colour), then a hue band around it.
    hist, _ = np.histogram(h[evidence], bins=180, range=(0, 180))
    cum = np.cumsum(hist)
    bg_hue = int(np.searchsorted(cum, max(1, cum[-1] // 2)))
    return _hue_near(h, bg_hue, 18) | (s < 30) | border

def _detect_bars_color(image_path: str | Path) -> list[dict[str, Any]]:
    """Detect bar regions via saturated-color segmentation (the robust,
    selective path).  Only pixels with real color saturation are kept, which
    naturally excludes black/gray grid lines, axis text and labels.

    Orientation is inferred from the candidate geometry: vertical bars share a
    bottom baseline (height encodes the value), horizontal bars share a left
    (or right) edge and vary in width.  Each returned bar carries an
    ``orientation`` field so downstream steps can dispatch correctly.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    H, W = img.shape[:2]

    bg = _estimate_background_mask(hsv)
    fg = ((s > 50) & (v > 50) & (~bg)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, 8)

    candidates: list[dict[str, Any]] = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if y < 100 or w < 25 or hh < 3:
            continue
        if x < 5 or y < 5 or x + w > W - 5 or y + hh > H - 5:
            continue
        region = fg[y : y + hh, x : x + w]
        if region.sum() < area * 0.2:
            continue
        candidates.append({"x": int(x), "y": int(y), "w": int(w), "h": int(hh)})

    orientation = _classify_bar_orientation(candidates)
    if orientation == "horizontal":
        candidates = _keep_horizontal_bars(candidates)
    elif orientation == "vertical":
        candidates = _keep_vertical_bars(candidates)
    for b in candidates:
        b["orientation"] = orientation
    return candidates


def _detect_bars_contrast(image_path: str | Path) -> list[dict[str, Any]]:
    """Detect bars by luminance contrast against the background.

    Fallback for light-gray / white bars on dark (or mid-tone) panels, which
    have little or no color saturation and are invisible to the color path.
    Unlike the earlier unconditional fusion, this path only runs when the
    color path found too few bars, and it keeps strict filters: a larger
    morphology kernel, a resolution-scaled minimum thickness (kills grid-line
    slivers), a solid fill ratio, and the same baseline/edge alignment checks.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    H, W = img.shape[:2]

    # Background luminance comes from the frame border (which is almost always
    # background), NOT from _estimate_background_mask: that mask marks every
    # low-saturation pixel as background, which would erase exactly the
    # light-gray/white bars this fallback is meant to recover.
    border = np.zeros(v.shape, dtype=bool)
    band_w = max(8, W // 40)
    band_h = max(8, H // 40)
    border[:band_h, :] = True
    border[-band_h:, :] = True
    border[:, :band_w] = True
    border[:, -band_w:] = True
    bg_v = float(np.median(v[border])) if border.sum() > 50 else 255.0
    contrast = (
        (np.abs(v.astype(np.int16) - bg_v) > max(28.0, 0.12 * bg_v)) & (v > 20)
    ).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    fg = cv2.morphologyEx(contrast, cv2.MORPH_CLOSE, kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, 8)

    min_thickness = max(12.0, H * 0.02)
    raw_candidates: list[dict[str, Any]] = []
    for i in range(1, n):
        x, y, w, hh, area = stats[i]
        if y < 80 or w < 25 or hh < min_thickness:
            continue
        if x < 5 or y < 5 or x + w > W - 5 or y + hh > H - 5:
            continue
        region = fg[y : y + hh, x : x + w]
        if region.sum() < area * 0.3:
            continue
        raw_candidates.append({"x": int(x), "y": int(y), "w": int(w), "h": int(hh)})

    split_candidates: list[dict[str, Any]] = []
    for b in raw_candidates:
        pieces = _split_merged_bar(fg, b["x"], b["y"], b["w"], b["h"])
        split_candidates.extend(pieces)

    orientation = _classify_bar_orientation(split_candidates)
    if orientation == "horizontal":
        candidates = _keep_horizontal_bars(split_candidates)
        # Recover an unaligned but bar-shaped component dropped by the shared
        # left-edge filter: a highlighted bar is often merged with its
        # category label, so its blob starts left of the axis.  Clip it back
        # to the common left edge and keep it when its thickness matches.
        if candidates:
            thicknesses = [b["h"] for b in candidates]
            med_t = float(np.median(thicknesses))
            edges = [b["x"] for b in candidates]
            edge = max(set(edges), key=edges.count)
            kept_ids = {id(b) for b in candidates}
            for b in split_candidates:
                if id(b) in kept_ids:
                    continue
                if not (med_t * 0.55 <= b["h"] <= med_t * 2.0):
                    continue
                if b["w"] < 3.0 * b["h"]:
                    continue
                if b["x"] < edge - 2 and b["x"] + b["w"] > edge:
                    b = {**b, "x": edge, "w": b["x"] + b["w"] - edge}
                candidates.append(b)
            candidates = _dedupe_y_overlap(candidates)
    elif orientation == "vertical":
        candidates = _keep_vertical_bars(split_candidates)
    for b in candidates:
        b["orientation"] = orientation
    return candidates


def detect_bars(image_path: str | Path) -> list[dict[str, Any]]:
    """Detect bar regions, preferring the selective color path.

    The color path is robust for ordinary charts (saturated bars on a light
    background) and never mistakes grid lines / text for bars.  It is only
    replaced by the luminance-contrast fallback when it finds too few bars
    (e.g. light-gray bars on a dark panel), so noisy contrast signals cannot
    poison charts the color path handles correctly.
    """
    color_bars = _detect_bars_color(image_path)
    if len(color_bars) >= 3:
        return color_bars
    try:
        contrast_bars = _detect_bars_contrast(image_path)
    except Exception:
        contrast_bars = []
    if len(contrast_bars) > len(color_bars):
        return contrast_bars
    return color_bars

def _filter_consistent_width(
    candidates: list[dict[str, Any]],
    frame_width: int,
) -> list[dict[str, Any]]:
    """Keep vertical bars that share a similar width and drop stray text
    blobs.  Real bars in one chart use the same category width; value labels
    and other text have irregular, often much narrower or wider boxes."""
    if len(candidates) < 3:
        return candidates
    # The reference width must come from bar-like (tall) components; a frame
    # full of text blobs would otherwise pull the median down and filter out
    # the real (wider) bars.
    ref = [b for b in candidates if b["h"] >= 40] or candidates
    widths = np.array([b["w"] for b in ref])
    med = float(np.median(widths))
    if med <= 0 or med < frame_width * 0.01:
        return candidates
    kept = [b for b in candidates if med * 0.45 <= b["w"] <= med * 2.0]
    return kept if len(kept) >= 2 else candidates


def _runs_above(proj: np.ndarray, threshold: int) -> list[tuple[int, int]]:
    """Return (start, end) runs where ``proj > threshold``."""
    runs: list[tuple[int, int]] = []
    start = -1
    for idx, val in enumerate(proj):
        if val > threshold and start < 0:
            start = idx
        elif val <= threshold and start >= 0:
            runs.append((start, idx))
            start = -1
    if start >= 0:
        runs.append((start, len(proj)))
    return runs


def _split_merged_bar(
    fg: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
) -> list[dict[str, Any]]:
    """Split a merged bar component into individual bars using projection gaps.

    Vertical bars sit side by side (gaps appear in the column projection),
    horizontal bars stack top-to-bottom (gaps appear in the row projection).
    Whichever direction yields multiple solid segments splits the blob.
    """
    region = fg[y : y + h, x : x + w] > 0
    col = region.sum(axis=0)
    row = region.sum(axis=1)
    col_thresh = max(2.0, 0.15 * float(col.max())) if col.size else 0.0
    row_thresh = max(2.0, 0.15 * float(row.max())) if row.size else 0.0
    col_runs = _runs_above(col, int(col_thresh))
    if len(col_runs) >= 2:
        pieces = []
        for s, e in col_runs:
            if e - s >= 8:
                pieces.append({"x": int(x + s), "y": int(y), "w": int(e - s), "h": int(h)})
        if len(pieces) >= 2:
            return pieces
    row_runs = _runs_above(row, int(row_thresh))
    pieces = []
    for s, e in row_runs:
        if e - s >= 8:
            pieces.append({"x": int(x), "y": int(y + s), "w": int(w), "h": int(e - s)})
    return pieces or [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)}]


def _classify_bar_orientation(candidates: list[dict[str, Any]]) -> str:
    """Decide whether candidate components look like vertical or horizontal
    bars: vertical bars share a bottom baseline; horizontal bars share a left
    (or right) edge and vary in width."""
    if len(candidates) < 2:
        return "vertical"
    tall = [b for b in candidates if b["h"] >= 25]
    if len(tall) >= 2:
        bottoms = np.array([b["y"] + b["h"] for b in tall])
        if float(np.ptp(bottoms)) <= 30:
            return "vertical"
    wide = [b for b in candidates if b["w"] >= 25 and 8 <= b["h"] <= 150]
    if len(wide) >= 2:
        for key in ("left", "right"):
            values = np.array([b["x"] if key == "left" else b["x"] + b["w"] for b in wide])
            rounded = np.round(values / 10.0).astype(np.int64)
            counts = np.bincount(rounded - rounded.min())
            peak = rounded.min() + int(np.argmax(counts))
            members = [b for b in wide if abs(round((b["x"] if key == "left" else b["x"] + b["w"]) / 10.0) - peak) <= 2]
            if len(members) >= 2:
                widths = np.array([b["w"] for b in members])
                if float(np.ptp(widths)) >= max(20.0, 0.2 * float(widths.max())):
                    return "horizontal"
    return "vertical"


def _keep_vertical_bars(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep components sitting on the dominant bottom baseline (vertical bars),
    dropping value circles / label text / other floating components."""
    if not candidates:
        return []
    tall = [b for b in candidates if b["h"] >= 25]
    if tall:
        # Use the most common bottom position (rounded to 10 px) as the
        # baseline instead of the median: a median can be pulled away by a
        # couple of non-bar components (e.g. value circles above bars).
        bottoms = np.array([b["y"] + b["h"] for b in tall])
        rounded = np.round(bottoms / 10.0).astype(np.int64)
        counts = np.bincount(rounded - rounded.min())
        peak = rounded.min() + int(np.argmax(counts))
        baseline = float(peak * 10)
        candidates = [
            b for b in candidates if abs((b["y"] + b["h"]) - baseline) <= 15
        ]
    candidates.sort(key=lambda b: b["x"])
    return candidates

def _keep_horizontal_bars(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep components that form horizontal bars: the majority share a common
    left (or right) edge and have a similar thickness; sort top-to-bottom."""
    if not candidates:
        return []
    wide = [b for b in candidates if b["w"] >= 25 and 8 <= b["h"] <= 150]
    if not wide:
        return []
    kept = []
    for key in ("left", "right"):
        values = np.array([b["x"] if key == "left" else b["x"] + b["w"] for b in wide])
        rounded = np.round(values / 10.0).astype(np.int64)
        counts = np.bincount(rounded - rounded.min())
        peak = rounded.min() + int(np.argmax(counts))
        members = [b for b in wide if abs(round((b["x"] if key == "left" else b["x"] + b["w"]) / 10.0) - peak) <= 2]
        if len(members) > len(kept):
            kept = members
    if len(kept) < 2:
        return []
    # Similar thickness (tolerant: labels may merge with bars, making one bar
    # noticeably thicker), then drop non-bar fragments that are far shorter
    # than the longest bar.
    thickness = np.array([b["h"] for b in kept])
    med = float(np.median(thickness))
    kept = [b for b in kept if med * 0.5 <= b["h"] <= med * 2.5]
    maxw = max((b["w"] for b in kept), default=0)
    if maxw > 0:
        kept = [b for b in kept if b["w"] >= max(25.0, 0.12 * maxw)]
    kept.sort(key=lambda b: b["y"])
    return kept

def _merge_vertical_neighbors(
    bars: list[dict[str, Any]],
    gap: int = 15,
) -> list[dict[str, Any]]:
    """Merge components that are vertically adjacent with the same left edge
    and width.  A bar whose top strip has a different (e.g. lighter) fill can
    be split into two blobs; merging them back yields one bar."""
    bars = sorted(bars, key=lambda b: (b["x"], b["y"]))
    merged: list[dict[str, Any]] = []
    for b in bars:
        if (
            merged
            and abs(b["x"] - merged[-1]["x"]) <= 2
            and abs(b["w"] - merged[-1]["w"]) <= 2
            and b["y"] - (merged[-1]["y"] + merged[-1]["h"]) <= gap
        ):
            prev = merged[-1]
            prev["h"] = b["y"] + b["h"] - prev["y"]
        else:
            merged.append(dict(b))
    return merged


def _y_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True when two horizontal-bar candidates overlap in the y range by more
    than half of the shorter one.  Real horizontal bars stack top-to-bottom
    without overlap; an overlapping blob is a value label / decoration of one
    of the bars (e.g. a label rendered just outside the bar's right end)."""
    a0, a1 = a["y"], a["y"] + a["h"]
    b0, b1 = b["y"], b["y"] + b["h"]
    inter = min(a1, b1) - max(a0, b0)
    return inter > 0.5 * min(a["h"], b["h"])


def _dedupe_y_overlap(bars: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop smaller blobs that overlap a retained bar in the y range."""
    bars = sorted(bars, key=lambda b: (b["y"], -b["h"]))
    kept: list[dict[str, Any]] = []
    for b in bars:
        if any(_y_overlap(b, k) for k in kept):
            continue
        kept.append(b)
    return kept


def _bar_length_vector(bars: list[dict[str, Any]]) -> list[float] | None:
    """Normalized bar lengths in categorical order (height for vertical bars,
    width for horizontal bars), divided by the longest bar."""
    if not bars:
        return None
    lengths = [_bar_length(b) for b in bars]
    max_len = max(lengths)
    if max_len <= 0:
        return None
    return [length / max_len for length in lengths]


def _frame_sharpness(path: str | Path) -> float:
    try:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return 0.0
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _segment_bar_plateaus(
    observations: list[dict[str, Any]],
    *,
    sample_fps: float,
    min_plateau_seconds: float,
    length_tolerance: float,
) -> list[list[dict[str, Any]]]:
    """Cut a bar-geometry time series into steady plateaus.

    A plateau is a maximal run where the bar count and every normalized
    length stay within ``length_tolerance``, lasting at least
    ``min_plateau_seconds``.  Frames with no detectable bars break the run;
    transitions (changing lengths) never form a state.
    """
    raw: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for obs in observations:
        if obs["vector"] is None:
            if current:
                raw.append(current)
                current = []
            continue
        if not current:
            current = [obs]
            continue
        prev = current[-1]
        same_count = obs["bar_count"] == prev["bar_count"]
        same_shape = same_count and all(
            abs(a - b) <= length_tolerance
            for a, b in zip(obs["vector"], prev["vector"])
        )
        if same_shape:
            current.append(obs)
        else:
            raw.append(current)
            current = [obs]
    if current:
        raw.append(current)
    kept = [
        run
        for run in raw
        if run[-1]["timestamp"] - run[0]["timestamp"] + 1.0 / sample_fps >= min_plateau_seconds
    ]
    # Merge adjacent plateaus with the same count and shape: a brief dip
    # below the minimum duration (e.g. a one-frame detection glitch) should
    # not split one steady state into two.
    def _same_shape(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
        a, b = left[-1], right[0]
        if a["bar_count"] != b["bar_count"] or a["vector"] is None or b["vector"] is None:
            return False
        return all(abs(x - y) <= length_tolerance for x, y in zip(a["vector"], b["vector"]))

    merged: list[list[dict[str, Any]]] = []
    for run in kept:
        if merged and _same_shape(merged[-1], run):
            merged[-1] = [*merged[-1], *run]
        else:
            merged.append(run)
    return merged


def detect_bar_states(
    video: str | Path,
    cfg: dict[str, Any] | None = None,
    *,
    sample_fps: float = 2.0,
    min_plateau_seconds: float = 0.8,
    length_tolerance: float = 0.06,
    min_bars: int = 2,
    expected_bar_count: int | None = None,
    out_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Segment a bar clip into steady states from bar geometry over time.

    A state is a plateau with the FULL bar count (``expected_bar_count``,
    normally the keyframe's detected count) holding the same normalized
    lengths for at least ``min_plateau_seconds``.  Rise/fall transitions and
    partial-reveal stages (fewer bars appearing one by one) never form a
    state, and the VLM's state/year labels are deliberately ignored (they
    are frequently hallucinated, e.g. bar_74's invented "2019").
    """
    cfg = cfg or {}
    state_cfg = cfg.get("state_detection") or {}
    sample_fps = float(state_cfg.get("sample_fps", sample_fps))
    min_plateau_seconds = float(state_cfg.get("min_plateau_seconds", min_plateau_seconds))
    length_tolerance = float(state_cfg.get("length_tolerance", length_tolerance))
    min_bars = int(state_cfg.get("min_bars", min_bars))
    expected_bar_count = int(expected_bar_count or state_cfg.get("expected_bar_count") or 0) or None
    out_dir = ensure_dir(out_dir) if out_dir is not None else None

    scan_root = (
        ensure_dir(Path(cfg.get("processed_root", "data/processed")) / "state_scan")
        if out_dir is None
        else ensure_dir(out_dir / "state_scan")
    )
    duration = float(ffprobe(video)["format"]["duration"])
    frames = extract_frames(video, scan_root, sample_fps, 768, "state_scan", force=True)

    observations = []
    for frame in frames:
        try:
            bars = detect_bars(frame["path"])
        except Exception:
            bars = []
        vector = _bar_length_vector(bars) if len(bars) >= min_bars else None
        observations.append(
            {
                "timestamp": round(float(frame["timestamp"]), 3),
                "path": str(frame["path"]),
                "bar_count": len(bars) if vector is not None else 0,
                "vector": vector,
                "sharpness": round(_frame_sharpness(frame["path"]), 3),
            }
        )

    plateaus = _segment_bar_plateaus(
        observations,
        sample_fps=sample_fps,
        min_plateau_seconds=min_plateau_seconds,
        length_tolerance=length_tolerance,
    )
    full_count = expected_bar_count
    if full_count is None:
        counts = [obs["bar_count"] for obs in observations if obs["bar_count"] > 0]
        full_count = max(counts) if counts else 0
    plateaus = [run for run in plateaus if run[0]["bar_count"] == full_count]

    states: list[dict[str, Any]] = []
    for idx, run in enumerate(plateaus, start=1):
        duration_plateau = run[-1]["timestamp"] - run[0]["timestamp"] + 1.0 / sample_fps
        # Representative: the sharpest settled frame, preferring one at least
        # 40% into the plateau (away from the entry edge).
        start_idx = max(0, min(len(run) - 1, int(len(run) * 0.4)))
        candidate = max(run[start_idx:], key=lambda obs: obs["sharpness"])
        representative_path = candidate["path"]
        if out_dir is not None:
            target = out_dir / "state_frames" / f"state_{idx:02d}.png"
            extract_still(video, candidate["timestamp"], target, force=True)
            representative_path = str(target)
        states.append(
            {
                "state_id": f"state_{idx:02d}",
                "start": run[0]["timestamp"],
                "end": run[-1]["timestamp"],
                "duration": round(duration_plateau, 3),
                "bar_count": run[0]["bar_count"],
                "length_vector": [round(value, 4) for value in run[0]["vector"]],
                "full_bar_count": full_count,
                "representative_timestamp": candidate["timestamp"],
                "representative_path": representative_path,
                "sample_count": len(run),
            }
        )
    if out_dir is not None:
        write_json(
            out_dir / "state_scan_report.json",
            {
                "clip_duration": round(duration, 3),
                "sample_fps": sample_fps,
                "full_bar_count": full_count,
                "states": states,
                "observations": observations,
            },
        )
    return states


def _clean_vision_label(part: str) -> str:
    part = str(part).strip().strip("*。·\t ")
    part = re.sub(r"^\d+[.)、:：]\s*", "", part)
    return part.strip()


def _normalize_label(text: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "", str(text).lower())
    aliases = {
        "eu": "europeanunion",
        "usa": "unitedstates",
        "us": "unitedstates",
        "uk": "unitedkingdom",
    }
    return aliases.get(normalized, normalized)


def _labels_match(want: str, label: str) -> bool:
    """Match two raw labels by token sets (case-insensitive).

    Substring matching is too loose: "insuredunitedstates" is a substring of
    "uninsuredunitedstates", so a recovered "Insured United States" swallows
    the "Uninsured United States" bar. Token-set matching keeps such labels
    distinct while still tolerating minor OCR noise (extra/duplicated words).
    Tokens must be extracted from the raw labels (before normalization strips
    the spaces), otherwise "insuredunitedstates" collapses into one token.
    """
    want_tokens = set(re.findall(r"[a-z0-9]+", str(want).lower()))
    label_tokens = set(re.findall(r"[a-z0-9]+", str(label).lower()))
    if not want_tokens or not label_tokens:
        return False
    return want_tokens <= label_tokens or label_tokens <= want_tokens


def _looks_like_value_label(text: str) -> bool:
    """Whether a vision "label" is really a printed VALUE misread as a
    category (e.g. "43,000", "36.1%", "43000") rather than a numeric
    category such as a year ("2019").  Years are bare 2-4 digit tokens;
    values carry group separators, decimals, unit symbols, or are longer.
    """
    token = str(text or "").strip()
    if not token:
        return True
    if re.search(r"[,\.%$]", token):
        return True
    if re.fullmatch(r"[0-9]{5,}", token):
        return True
    return False


def _labeled_value_pairs(text: str) -> list[tuple[str, str]]:
    """Extract ``Label: value`` pairs from a vision response."""
    pairs: list[tuple[str, str]] = []
    for match in re.finditer(r"([A-Za-z][A-Za-z0-9 $&'\-\.\,]*?):\s*(-?\d+(?:,\d{3})*(?:\.\d+)?\s*%?)", text):
        label = _clean_vision_label(match.group(1))
        value_match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?\s*%?", match.group(2))
        if label and value_match:
            pairs.append((label, value_match.group(0).strip()))
    return pairs


def _parse_label_json(text: str) -> list[str]:
    """Extract a JSON array of label strings (or {"label": ...} objects) from
    a vision response, tolerating ```json fences and prose."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        parsed = json.loads(cleaned[start : end + 1])
    except Exception:
        return []
    labels: list[str] = []
    for item in parsed if isinstance(parsed, list) else []:
        if isinstance(item, str):
            labels.append(item)
        elif isinstance(item, dict) and item.get("label"):
            labels.append(str(item["label"]))
    return [str(label).strip() for label in labels if str(label).strip()]


def match_entities(
    boxes: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    vision_order: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Match boxes to entities by categorical order (vision-confirmed if
    available).

    Pass 1 matches labels (including numeric categories such as years) from
    the vision order against the recovered entities.  Pass 2 assigns the
    first still-unmatched entity to any bar whose label is missing or
    unreadable (e.g. a clipped brand logo), so the bar keeps the recovered
    entity name instead of inventing a new one.  A label that names no
    recovered entity creates a new frame entity only when it is a real
    category (years are fine); a value-like token (e.g. "43,000") is a
    misread and never becomes an entity.
    """
    aligned: list[dict[str, Any] | None] = [None] * len(boxes)
    warnings: list[str] = []
    used_entities: set[int] = set()
    pending: list[tuple[int, str]] = []

    for i, box in enumerate(boxes):
        want = ""
        if vision_order and i < len(vision_order):
            want = _clean_vision_label(vision_order[i])
        norm_want = _normalize_label(want)
        entity = None
        if norm_want:
            for ei, e in enumerate(entities):
                if ei in used_entities:
                    continue
                label = _normalize_label(e.get("label"))
                if label == norm_want or (label and _labels_match(want, str(e.get("label") or ""))):
                    entity = e
                    used_entities.add(ei)
                    break
        if entity is None:
            pending.append((i, want))
            continue
        aligned[i] = (
            {
                **box,
                "entity_id": entity["id"],
                "label": entity["label"],
                "entity_source": entity.get("entity_source", "recovered"),
            }
        )

    unused = [ei for ei in range(len(entities)) if ei not in used_entities]
    for i, want in pending:
        box = boxes[i]
        entity = None
        if unused:
            entity = entities[unused.pop(0)]
        elif not _looks_like_value_label(want):
            # A label that names an entity the recovered table missed.
            entity = {
                "id": re.sub(r"[^a-z0-9]+", "-", want.lower()).strip("-") or f"bar-{i + 1}",
                "label": want,
                "entity_source": "frame",
            }
            warnings.append(f"created entity from frame label: {want}")
        if entity is None:
            warnings.append(f"no entity for box #{i + 1} at x={box['x']}")
            continue
        aligned[i] = (
            {
                **box,
                "entity_id": entity["id"],
                "label": entity["label"],
                "entity_source": entity.get("entity_source", "recovered"),
            }
        )
    if len(boxes) > len(entities):
        warnings.append(f"detected {len(boxes)} bars but only {len(entities)} entities recovered")
    return [a for a in aligned if a is not None], warnings


def read_entity_order(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
    orientation: str = "vertical",
) -> list[str]:
    """Ask the vision model for the category labels of the bars, ordered along
    the categorical axis (left-to-right for vertical bars, top-to-bottom for
    horizontal bars)."""
    if orientation == "horizontal":
        prompt = (
            "这是横向条形图的一帧。请从上到下依次列出每根条形左侧的类别名称。"
            '只返回 JSON 数组，格式 [{"label":"类别名"}, ...]，'
            '类别名必须保持完整（例如 "Less than $20,000"，里面的逗号是数字分隔符，绝不能拆分），不要额外解释。'
        )
    else:
        prompt = (
            "这是柱状图的一帧。请从左到右依次列出每根柱子底部的类别标签，"
            "只要名字本身，用逗号分隔，不要额外解释。"
        )
    text = _call_vision(
        image_path,
        prompt,
        cfg,
        temperature=0.0,
    )
    if orientation == "horizontal":
        json_labels = _parse_label_json(text)
        if json_labels:
            return [_clean_vision_label(label) for label in json_labels]
    labels = []
    skip = {"从左到右", "依次为", "第一根", "第二根", "第三根", "第四根", "柱子", "类别", "标签"}
    for part in re.split(r"[,，\n;；、]", text):
        part = _clean_vision_label(part)
        if part and part.lower() not in skip:
            labels.append(part)
    return labels


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced ``{...}`` JSON object from a model response."""
    text = str(text or "")
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : idx + 1])
                except (TypeError, ValueError):
                    return None
    return None


def read_chart_table(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
    orientation: str = "vertical",
    attempts: int = 2,
) -> dict[str, Any]:
    """Read the complete data table from the keyframe via the vision model.

    This is the authoritative data source for the keyframe state: it returns
    every bar's category label and printed value in categorical order, plus
    the in-frame title and unit symbol.  A value is only marked ``verified``
    when at least two independent reads agree, so a single hallucinated
    number falls back to the per-bar crop read / scale estimation downstream.
    """
    if orientation == "horizontal":
        prompt = (
            "这是横向条形图的一帧。请完整读出图表数据，只返回一个 JSON 对象，"
            "不要任何解释或 markdown：{\"title\": \"图表标题原文，读不准就空字符串\", "
            "\"unit\": \"数值单位符号（如 $、%、k、M；没有就空字符串，禁止臆测）\", "
            "\"bars\": [{\"label\": \"条形名称\", \"value\": 数字}]}。"
            "要求：1) 按从上到下顺序列出每根条形；2) value 必须是画面中实际印刷的数字，"
            "保留完整大小（37,000 写成 37000）；3) 数值可能印在条形内部、右端、"
            "或条形名称旁边（如名称正下方的数据标签），只要属于该条形行的印刷数字都算；"
            "4) 坐标轴刻度（如 0/10,000）不是条形数值，绝不能写进 bars；"
            "5) 完全没有印刷数值的条形省略 value 或填 null，绝不能编造；"
            "6) 标签读不准就填空字符串，绝不能编造。"
        )
    else:
        prompt = (
            "这是柱状图的一帧。请完整读出图表数据，只返回一个 JSON 对象，"
            "不要任何解释或 markdown：{\"title\": \"图表标题原文，读不准就空字符串\", "
            "\"unit\": \"数值单位符号（如 $、%、k、M；没有就空字符串，禁止臆测）\", "
            "\"bars\": [{\"label\": \"柱子底部名称\", \"value\": 数字}]}。"
            "要求：1) 按从左到右顺序列出每根柱子；2) value 必须是画面中实际印刷的数字，"
            "保留完整大小（37,000 写成 37000）；3) 数值可能印在柱子内部、顶部、"
            "或柱子名称旁边（如名称正上方的数据标签），只要属于该柱子列的印刷数字都算；"
            "4) 坐标轴刻度（如 0/10,000）不是柱子数值，绝不能写进 bars；"
            "5) 完全没有印刷数值的柱子省略 value 或填 null，绝不能编造；"
            "6) 标签读不准就填空字符串，绝不能编造。"
        )
    raw_attempts: list[str] = []
    row_sets: list[list[dict[str, Any]]] = []
    titles: list[str] = []
    units: list[str] = []
    for _ in range(max(1, attempts)):
        try:
            text = _call_vision(image_path, prompt, cfg, temperature=0.0)
        except Exception:
            continue
        raw_attempts.append(text)
        obj = _extract_json_object(text)
        if not isinstance(obj, dict):
            continue
        title = str(obj.get("title") or "").strip()
        unit = str(obj.get("unit") or "").strip()
        if title:
            titles.append(title)
        if unit:
            units.append(unit)
        rows: list[dict[str, Any]] = []
        for bar in obj.get("bars") if isinstance(obj.get("bars"), list) else []:
            if not isinstance(bar, dict):
                continue
            label = str(bar.get("label") or "").strip()
            value = bar.get("value")
            value_num: float | None = None
            value_text = ""
            if value not in (None, ""):
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    value_num = float(value)
                    value_text = f"{value_num:g}"
                else:
                    match = _NUMBER_TOKEN_RE.search(str(value))
                    if match:
                        token = match.group(0).strip()
                        value_text = token
                        try:
                            value_num = float(token.replace(",", "").rstrip("%"))
                        except ValueError:
                            value_num = None
            rows.append({"label": label, "value": value_num, "value_text": value_text})
        row_sets.append(rows)

    count = max((len(rows) for rows in row_sets), default=0)
    merged: list[dict[str, Any]] = []
    for idx in range(count):
        candidates = [rows[idx] for rows in row_sets if idx < len(rows)]
        if not candidates:
            break
        labels = {c["label"] for c in candidates if c["label"]}
        label = sorted(labels)[0] if len(labels) == 1 else (next(iter(labels)) if labels else "")
        value: float | None = None
        value_text = ""
        agreeing = 0
        if len({c["value"] for c in candidates if c.get("value") is not None}) == 1:
            value = next((c["value"] for c in candidates if c.get("value") is not None), None)
            agreeing = sum(1 for c in candidates if c.get("value") == value)
            for c in candidates:
                if c.get("value") == value and c.get("value_text"):
                    value_text = c["value_text"]
                    break
        verified = agreeing >= 2
        merged.append(
            {
                "label": label,
                "value": value,
                "value_text": value_text,
                "verified": verified,
            }
        )
    return {
        "title": titles[0] if titles else "",
        "unit": units[0] if units else "",
        "bars": merged,
        "attempt_count": len(row_sets),
        "raw_attempts": raw_attempts,
    }


def read_bar_label(
    image_path: str | Path,
    box: dict[str, Any],
    cfg: dict[str, Any] | None = None,
) -> str:
    """Read one bar's category label from a focused crop (left of a horizontal
    bar, below a vertical bar).  Used when the full-frame table read missed a
    small / clipped label."""
    img = cv2.imread(str(image_path))
    if img is None:
        return ""
    H, W = img.shape[:2]
    if str(box.get("orientation")) == "horizontal":
        # Category text sits to the LEFT of a horizontal bar; some charts also
        # print it just above the bar, so include a band above as well.
        x2 = max(10, int(box["x"]) - 2)
        x1 = max(0, x2 - 240)
        y1 = max(0, int(box["y"]) - 28)
        y2 = min(H, int(box["y"] + box["h"]) + 10)
    else:
        y1 = int(box["y"] + box["h"]) + 2
        y2 = min(H, y1 + 70)
        x1 = max(0, int(box["x"]) - 90)
        x2 = min(W, int(box["x"] + box["w"]) + 90)
    if y2 <= y1:
        y2 = y1 + 1
    if x2 <= x1:
        x2 = x1 + 1
    crop = img[y1:y2, x1:x2]
    crop_path = Path(image_path).with_name(f"label_crop_{int(box['x'])}_{int(box['y'])}.png")
    cv2.imwrite(str(crop_path), crop)
    try:
        text = _call_vision(crop_path, "读出这个裁剪图里的文字，只返回文字本身，不要解释。", cfg, temperature=0.0)
    except Exception:
        return ""
    label = text.strip().strip("\"'`。，,. ")
    if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", label):
        return ""
    return label[:60] if len(label) > 60 else label


def _assign_table_values(
    aligned: list[dict[str, Any]],
    table: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge verified table values into the CV-aligned bars.

    Matches by label first (normalized), then by categorical position, so a
    bar whose label the table read with a slight spelling difference still
    receives its verified printed value.
    """
    rows = table.get("bars") or []
    by_label: dict[str, dict[str, Any]] = {}
    for row in rows:
        norm = _normalize_label(row.get("label"))
        if norm:
            by_label.setdefault(norm, row)
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(aligned):
        row = None
        norm = _normalize_label(item.get("label"))
        if norm and norm in by_label:
            row = by_label[norm]
        elif idx < len(rows):
            row = rows[idx]
        if row and row.get("verified") and row.get("value") is not None:
            out.append(
                {
                    **item,
                    "value": row["value"],
                    "value_text": row.get("value_text") or "",
                    "value_read_verified": True,
                }
            )
        else:
            out.append({**item, "value": None, "value_text": None, "value_read_verified": False})
    return out


def read_frame_title(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Read the chart title printed in the frame via the vision model.

    Used when the VLM data recovery missed the title (its visible_text has no
    title candidate at all, e.g. bar_29 "Monthly price of Advair asthma
    inhaler" while the recovered title was the video title).
    """
    text = _call_vision(
        image_path,
        "读出这张图表顶部的主标题文字（忽略副标题和数据来源行），只返回标题本身。"
        "如果图表没有主标题，只返回 NONE，不要解释或描述。",
        cfg,
        temperature=0.0,
    )
    title = text.strip().strip("\"'`。，,.")
    if title.upper() in {"NONE", "无", "没有", "无标题", "NO TITLE"} or len(title) > 120:
        return ""
    if len(title) > 120:
        title = title[:120]
    return title


def verify_alignment(image_path: str | Path, cfg: dict[str, Any] | None = None) -> tuple[bool, str]:
    """Ask the vision model whether the overlaid labels match the bars."""
    text = _call_vision(
        image_path,
        "检查这张图上叠加的文字标签是否与柱子的真实归属匹配（例如最高的柱子上写的是哪个实体名，它应该对应柱底的真实标签）。"
        "用一句话回答：匹配 或 不匹配，并说明原因。",
        cfg,
        temperature=0.0,
    )
    bad = any(k in text for k in ("不匹配", "错位", "不一致", "错误", "不符"))
    if bad:
        return False, text[:300]
    if any(k in text for k in ("匹配", "一致", "正确", "对应")):
        return True, text[:300]
    return False, text[:300]


def _crop_value_region(img: np.ndarray, box: dict[str, Any]) -> np.ndarray:
    if str(box.get("orientation")) == "horizontal":
        # Horizontal bars print the value at the right end (or just inside
        # the end).  Crop a band centred vertically on the bar, starting at
        # ~55% of the bar width (so values printed inside the bar tail are
        # included) and extending ~140px past the right edge (right-end
        # labels).  This is scale-agnostic: no absolute pixel thresholds.
        H, W = img.shape[:2]
        bar_x2 = int(box["x"] + box["w"])
        x1 = max(0, int(box["x"] + box["w"] * 0.55))
        x2 = min(W, bar_x2 + 140)
        y1 = max(0, int(box["y"]) - 6)
        y2 = min(H, int(box["y"] + box["h"]) + 6)
        if x2 <= x1:
            x2 = min(W, x1 + 140)
        if y2 <= y1:
            y1 = max(0, int(box["y"]) - 8)
            y2 = min(H, int(box["y"] + box["h"]) + 8)
        return img[y1:y2, x1:x2]
    x1 = max(0, box["x"] - 20)
    x2 = min(img.shape[1], box["x"] + box["w"] + 20)
    y1 = max(0, box["y"] - 60)
    y2 = min(img.shape[0], box["y"] + 8)
    if y2 <= y1 or x2 <= x1:
        return img[max(0, box["y"] - 40) : box["y"] + 8, x1:x2]
    return img[y1:y2, x1:x2]


def read_bar_values(
    image_path: str | Path,
    aligned: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
    orientation: str = "vertical",
) -> list[dict[str, Any]]:
    """Read each bar's printed value via one full-frame vision call.

    The vision model reads all printed values left-to-right, which is far more
    reliable than per-bar crops (a short bar's crop can include unrelated text).
    Bars with no value from the full-frame call fall back to a focused crop.
    """
    values: list[str | None] = [None] * len(aligned)
    if orientation == "horizontal":
        value_prompt = (
            "这是横向条形图。请找出画面中**实际印刷了数值**的条形，用「标签: 数值」的格式"
            "列出它们的左侧完整名称和右端数值（例如 Less than $20,000: 890）。"
            "注意：坐标轴上的刻度数字（如底部 x 轴的 0/100/200 或 $30,000）**不是条形数值**，"
            "绝不能当作条形的值；只读与条形相邻（内部/右端）的数值标签。"
            "没有印刷数值的条形**绝对不能写数值、绝对不能编造**，只列有数值的条形即可；"
            "宁可少报，不要瞎编。不要解释。"
        )
    else:
        value_prompt = (
            "这是柱状图的一帧。请找出画面中**实际印刷了数值**的柱子，用「标签: 数值」"
            "的格式列出它们的底部标签和顶部数值（例如 Sub-Saharan Africa: 36.1%）。"
            "注意：坐标轴上的刻度数字（如左侧 y 轴的 0/$30,000）**不是柱子数值**，"
            "绝不能当作柱子的值；只读与柱子相邻（上方/内部）的数值标签。"
            "没有印刷数值的柱子**绝对不能写数值、绝对不能编造**，只列有数值的柱子即可；"
            "宁可少报，不要瞎编。不要解释。"
        )
    attempts: list[str] = []
    for _ in range(3):
        try:
            text = _call_vision(image_path, value_prompt, cfg, temperature=0.0)
        except Exception:
            continue
        attempts.append(text)

    per_bar: list[list[str]] = [[] for _ in aligned]
    for text in attempts:
        assigned: set[int] = set()
        for label, value in _labeled_value_pairs(text):
            norm = _normalize_label(label)
            if not norm:
                continue
            for idx, item in enumerate(aligned):
                if idx in assigned:
                    continue
                item_label = _normalize_label(item.get("label"))
                if item_label and (norm in item_label or item_label in norm):
                    per_bar[idx].append(value)
                    assigned.add(idx)
                    break
        if not assigned and orientation != "horizontal":
            tokens = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?\s*%?", text)
            percentages = [token for token in tokens if "%" in token]
            sequence = percentages if len(percentages) >= len(aligned) else tokens
            for idx, item in enumerate(aligned):
                if idx >= len(sequence):
                    break
                per_bar[idx].append(sequence[idx])

    for idx, candidates in enumerate(per_bar):
        if candidates:
            majority = max(set(candidates), key=candidates.count)
            count = candidates.count(majority)
            # Only trust a value when at least two of three attempts agree;
            # a value seen once is likely a hallucination (e.g. an unlabeled
            # bar given a neighbour's number) and should be left for the
            # scale estimation instead.
            if len(attempts) >= 2 and count < 2:
                values[idx] = None
            else:
                values[idx] = majority

    img = cv2.imread(str(image_path))
    # A full-frame read that found no printed value at all is a strong signal
    # that this chart has no printed values (e.g. an unlabeled bar chart with
    # only an axis). Skip the per-bar crop fallback in that case: it would
    # burn one vision call per bar for nothing and the axis-tick estimation
    # should fill the values instead.
    any_full_frame_value = any(values)
    out: list[dict[str, Any]] = []
    for i, item in enumerate(aligned):
        value_text = values[i]
        if value_text is None and img is not None and any_full_frame_value:
            crop = _crop_value_region(img, item)
            crop_path = Path(image_path).with_name(f"value_crop_{i:02d}.png")
            cv2.imwrite(str(crop_path), crop)
            try:
                crop_text = _call_vision(
                    crop_path,
                    "读出这个裁剪图里的数字或百分比，只返回数字和单位，如 36.1% 或 6.9。",
                    cfg,
                    temperature=0.0,
                )
                match = _NUMBER_TOKEN_RE.search(crop_text)
                value_text = match.group(0).strip() if match else None
            except Exception:
                value_text = None
        out.append({**item, "value_text": value_text})
    return out


def _value_plausibility(item: dict[str, Any], aligned: list[dict[str, Any]]) -> tuple[bool, str]:
    """Check a frame-read value against the chart's printed-value conventions."""
    value = item.get("value")
    if value is None:
        return False, "no value"
    value_text = str(item.get("value_text") or "")
    if value < 0:
        return False, f"value {value:g} is negative"
    if "%" in value_text and value > 100:
        return False, f"value {value:g} outside plausible 0-100 range for percentages"
    if item.get("value_read_verified"):
        # Directly printed values are trusted over noisy bar-length geometry
        # (detection can under/over-measure a bar by several percent), so the
        # ratio check is skipped for them.
        return True, "ok"
    lengths = [_bar_length(a) for a in aligned if _bar_length(a) > 0]
    values = [
        float(a["value"])
        for a in aligned
        if isinstance(a.get("value"), (int, float)) and a["value"] > 0
    ]
    if len(lengths) >= 2 and len(values) >= 2:
        max_len = max(lengths)
        max_v = max(values)
        value_ratio = value / max_v
        length_ratio = _bar_length(item) / max_len
        tolerance = 0.15 if value < 5 else 0.12
        if abs(value_ratio - length_ratio) > tolerance:
            return False, (
                f"value ratio {value_ratio:.3f} vs bar length ratio {length_ratio:.3f} "
                "differs beyond tolerance"
            )
    return True, "ok"


def _ratio_consistency(aligned: list[dict[str, Any]]) -> tuple[bool, str]:
    if len(aligned) < 2:
        return True, "too few bars to check"
    lengths = [_bar_length(a) for a in aligned]
    max_len = max(lengths)
    if max_len <= 0:
        return True, "degenerate bar lengths"
    values = [
        float(a["value"])
        for a in aligned
        if isinstance(a.get("value"), (int, float)) and a.get("value")
    ]
    if not values:
        return True, "no numeric values to check"
    maxv = max(values)
    errors = []
    for a in aligned:
        v = a.get("value")
        if not isinstance(v, (int, float)) or v == 0:
            continue
        expected = _bar_length(a) / max_len
        val_ratio = float(v) / maxv
        if abs(val_ratio - expected) > 0.12:
            errors.append(f"{a['label']}: value ratio {val_ratio:.2f} vs bar length ratio {expected:.2f}")
    if errors:
        return False, "; ".join(errors)
    return True, "ok"


def _bar_length(item: dict[str, Any]) -> float:
    """The dimension that encodes the value: height for vertical bars, width
    for horizontal bars."""
    if str(item.get("orientation")) == "horizontal":
        return float(item.get("w") or 0.0)
    return float(item.get("h") or 0.0)


def estimate_unlabeled_values(
    aligned: list[dict[str, Any]],
    chart_type: str = "bar",
) -> int:
    """Estimate values for marks without printed values from the linear scale
    implied by the marks that do have printed values.

    The general rule: whatever geometric dimension encodes the value (bar
    length/height, line/point position along the value axis, pie arc angle),
    a linear calibration over the labeled marks maps that dimension back to
    values for the unlabeled ones.

    Currently implemented:
      * bar (vertical and horizontal): value ~ bar length (height or width).
    Estimated marks are flagged with ``value_estimated`` / ``value_type`` and
    carry a lower confidence + ``needs_review`` when reconciled.
    """
    if chart_type not in ("bar", "combined"):
        return 0
    labeled: list[tuple[float, float]] = []
    for item in aligned:
        value = item.get("value")
        length = _bar_length(item)
        if length > 0 and isinstance(value, (int, float)) and item.get("value_text"):
            labeled.append((length, float(value)))
    if len(labeled) < 2:
        return 0
    xs = np.array([x for x, _ in labeled], dtype=float)
    ys = np.array([y for _, y in labeled], dtype=float)
    try:
        slope, intercept = np.polyfit(xs, ys, 1)
    except Exception:
        return 0
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return 0
    count = 0
    for item in aligned:
        if item.get("value") is not None:
            continue
        length = _bar_length(item)
        if length <= 0:
            continue
        est = float(intercept + slope * length)
        if est < 0 or not np.isfinite(est):
            est = 0.0
        item["value"] = est
        item["value_text"] = f"{est:.0f}"
        item["value_estimated"] = True
        item["value_type"] = "estimated"
        item["plausibility_message"] = "estimated from labeled-bar scale"
        count += 1
    return count


def detect_axis_tick_marks(
    image_path: str | Path,
    orientation: str = "vertical",
) -> list[dict[str, Any]]:
    """Detect value-axis tick positions.

    Returns a list of ``{"coord": float}`` sorted along the value axis
    (``coord`` is the tick's y for vertical bars, x for horizontal bars).
    Colour-agnostic and layered so it works on bar charts with grid lines,
    line charts that only draw tick labels, and simple axis+short-stroke
    charts:
      Pass 1: grid lines / baseline (Canny edges, plot-region thin bands);
      Pass 2: short tick strokes next to a long axis line (contrast mask);
      Pass 3: tick label text blocks in the left/bottom margin.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    background = float(np.median(gray))
    edges = cv2.Canny(gray, 40, 120)
    contrast = (np.abs(gray.astype(np.int16) - background) > 25).astype(np.uint8) * 255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.int16)
    # Axis lines and tick strokes are near-neutral (black/white/gray);
    # saturated bars, shading bands and decorations must not feed the
    # axis+short-stroke detection.
    axis_contrast = np.where((contrast > 0) & (saturation < 60), 255, 0).astype(np.uint8)

    # Pass 1: thin lines spanning the plot (grid lines / baseline).
    if orientation == "horizontal":
        y_start = int(height * 0.12)
        y_end = int(height * 0.88)
        column_spans = []
        column_cnts = []
        for x in range(width):
            indices = np.where(edges[y_start:y_end, x] > 0)[0]
            if indices.size == 0:
                continue
            span = int(indices.max()) - int(indices.min())
            if span >= 0.45 * (y_end - y_start):
                column_spans.append(x)
                column_cnts.append(int(indices.size))
        bands = _cluster_consecutive(column_spans, gap=2)
        coords = []
        span_index = {x: cnt for x, cnt in zip(column_spans, column_cnts)}
        for band in bands:
            if len(band) > 8:
                continue
            counts = [span_index[x] for x in band]
            if sum(counts) / len(counts) < 25:
                continue
            coords.append(float(sum(band)) / len(band))
    else:
        x_start = int(width * 0.25)
        row_spans = []
        row_cnts = []
        for y in range(int(height * 0.15), height):
            indices = np.where(edges[y, x_start:] > 0)[0]
            if indices.size == 0:
                continue
            span = int(indices.max()) - int(indices.min())
            if span >= 0.4 * (width - x_start):
                row_spans.append(y)
                row_cnts.append(int(indices.size))
        bands = _cluster_consecutive(row_spans, gap=2)
        coords = []
        span_index = {y: cnt for y, cnt in zip(row_spans, row_cnts)}
        for band in bands:
            if len(band) > 8:
                continue
            counts = [span_index[y] for y in band]
            # Thin grid lines carry hundreds of dark pixels; sparse rows
            # (a few pixels) are noise or text remnants.
            if sum(counts) / len(counts) < 25:
                continue
            coords.append(float(sum(band)) / len(band))
    if coords:
        # A Canny grid line produces two edge rows ~5-8 px apart; merge them
        # so each tick maps to one coordinate.
        merged_coords: list[float] = []
        for coord in sorted(coords):
            if not merged_coords or coord - merged_coords[-1] > 10.0:
                merged_coords.append(coord)
            else:
                merged_coords[-1] = (merged_coords[-1] + coord) / 2.0
        series = _dominant_even_series(merged_coords)
        if len(series) >= 2:
            return [{"coord": coord} for coord in series]

    # Pass 2: short tick strokes next to a long axis line. Uses the contrast
    # mask (not Canny) so thin black/white axis lines are detected on any
    # background.
    if orientation == "horizontal":
        long_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        stroke_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 9))
        long_lines = cv2.morphologyEx(axis_contrast, cv2.MORPH_OPEN, long_kernel)
        strokes = cv2.morphologyEx(axis_contrast, cv2.MORPH_OPEN, stroke_kernel)
        row_scores = long_lines.sum(axis=1)
        axis_rows = [
            row
            for row in range(height)
            if row_scores[row] > width * 0.12
        ]
        if not axis_rows:
            return []
        axis_y = float(np.median(axis_rows))
        contours, _ = cv2.findContours(strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coords: list[float] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h < 6 or h > 34 or w > 12:
                continue
            if abs((y + h / 2) - axis_y) > 22:
                continue
            coords.append(x + w / 2)
    else:
        long_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        stroke_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2))
        long_lines = cv2.morphologyEx(axis_contrast, cv2.MORPH_OPEN, long_kernel)
        strokes = cv2.morphologyEx(axis_contrast, cv2.MORPH_OPEN, stroke_kernel)
        col_scores = long_lines.sum(axis=0)
        axis_cols = [
            col
            for col in range(width)
            if col_scores[col] > height * 0.12
        ]
        if not axis_cols:
            return []
        axis_x = float(np.median(axis_cols))
        contours, _ = cv2.findContours(strokes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coords: list[float] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 6 or w > 34 or h > 12:
                continue
            if abs((x + w / 2) - axis_x) > 22:
                continue
            coords.append(y + h / 2)

    coords = sorted(set(round(coord, 1) for coord in coords))
    merged: list[float] = []
    for coord in coords:
        if not merged or abs(coord - merged[-1]) > 6.0:
            merged.append(coord)
    if merged:
        series = _dominant_even_series(merged)
        if len(series) >= 2:
            return [{"coord": coord} for coord in series]

    label_centers = _detect_tick_label_blocks(img, orientation)
    if len(label_centers) >= 2:
        series = _dominant_even_series(label_centers)
        if len(series) >= 2:
            return [{"coord": coord} for coord in series]
    return []

def _detect_tick_label_blocks(
    img: np.ndarray,
    orientation: str = "vertical",
) -> list[float]:
    """Detect value-axis tick label text blocks and return their centres.

    Charts without grid lines often carry only a few sparse tick labels on
    the axis (e.g. "$350 / $175 / $0"); the label text itself is then the
    most reliable tick anchor. Labels are short text blocks; long lines (a
    shaded target-band boundary, axes, decorations) are rejected by the width
    cap. A horizontal closing pass joins the individual glyphs of one label
    ("50k" -> one block) so light-on-dark labels whose strokes are
    disconnected still register.

    The whole frame is scanned (no fixed left-margin assumption) and the
    blocks are clustered by position: value-axis labels form one regular
    column (vertical) or row (horizontal) of small blocks, while x-axis
    labels / legends / annotations sit elsewhere and drop out.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    background = float(np.median(gray))
    contrast = (np.abs(gray.astype(np.int16) - background) > 25).astype(np.uint8) * 255
    # Remove long straight lines (axes, grid lines) first: a label touching
    # the axis would otherwise merge into an over-wide block and be rejected.
    # Text strokes are short, so a long horizontal/vertical opening leaves
    # them intact while dropping the lines.
    long_h = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, np.ones((1, 61), np.uint8))
    long_v = cv2.morphologyEx(contrast, cv2.MORPH_OPEN, np.ones((61, 1), np.uint8))
    contrast = cv2.subtract(contrast, long_h)
    contrast = cv2.subtract(contrast, long_v)
    contrast = cv2.morphologyEx(contrast, cv2.MORPH_CLOSE, np.ones((1, 15), np.uint8))
    n_comp, _, stats, _ = cv2.connectedComponentsWithStats(contrast, 8)
    blocks: list[tuple[float, float, int, int]] = []  # (cx, cy, w, h)
    for i in range(1, n_comp):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if not (20 <= w <= 140 and 10 <= h <= 45):
            continue
        if area < w * 2 or area > w * 50:
            continue
        blocks.append((x + w / 2.0, y + h / 2.0, w, h))
    if len(blocks) < 2:
        return []
    if orientation == "horizontal":
        # x-axis labels: one horizontal row of blocks sharing a y band.
        rows: list[list[tuple[float, float]]] = []
        for block in sorted(blocks, key=lambda b: b[1]):
            cx, cy = block[0], block[1]
            if rows and abs(cy - rows[-1][-1][1]) <= 10:
                rows[-1].append((cx, cy))
            else:
                rows.append([(cx, cy)])
        best = max(rows, key=len)
        xs = [cx for cx, _ in best]
        if len(xs) < 2:
            return []
        return _dominant_even_series(sorted(xs), tolerance=0.25)
    # vertical: y-axis labels form one column of blocks sharing an x band.
    columns: list[list[tuple[float, float]]] = []
    for block in sorted(blocks, key=lambda b: b[0]):
        cx, cy = block[0], block[1]
        if columns and abs(cx - columns[-1][-1][0]) <= 60:
            columns[-1].append((cx, cy))
        else:
            columns.append([(cx, cy)])
    # A real value-axis label column spans a good share of the plot height;
    # stray annotations / title-card banners cover only a few pixels and are
    # dropped before ranking columns.
    min_span = 0.15 * height
    viable = [col for col in columns if (max(cy for _, cy in col) - min(cy for _, cy in col)) >= min_span]
    if not viable:
        return []
    best = max(viable, key=len)
    ys = [cy for _, cy in best]
    if len(ys) < 2:
        return []
    return _dominant_even_series(sorted(ys), tolerance=0.25)


def _dominant_even_series(values: list[float], tolerance: float = 0.45) -> list[float]:
    """Return the evenly-spaced subsequence covering the most points.

    Tick labels on an axis are equally spaced; stray text remnants (a broken
    glyph, a partial label) break the spacing and are excluded.
    """
    values = sorted(values)
    if len(values) < 3:
        return values
    best: list[float] = []
    for start_index in range(len(values)):
        for end_index in range(start_index + 1, len(values)):
            span = values[end_index] - values[start_index]
            count = end_index - start_index
            if span <= 0 or count < 2:
                continue
            step = span / count
            if step < 8:
                continue
            series: list[float] = []
            expected = values[start_index]
            for value in values:
                if abs(value - expected) <= tolerance * step:
                    series.append(value)
                    expected += step
            if len(series) > len(best):
                best = series
    return best

def _rebuild_even_ticks(detected: list[float], count: int, tolerance: float = 14.0) -> list[float]:
    """Rebuild a complete evenly-spaced tick set from noisy partial detections.

    Axis ticks are equally spaced and the vision-read label list gives the
    expected count. The median pairwise gap is the spacing; each detected
    position is scored as a candidate start and the one aligning with the
    most detections anchors the sequence ``start + k * spacing``.
    """
    points = sorted(float(value) for value in detected)
    if len(points) < 2 or count < 2:
        return points
    # Adjacent gaps only: pairwise gaps mix single-tick and multi-tick
    # spacing (a missing tick doubles the gap) and pollute the median.
    gaps = [points[index + 1] - points[index] for index in range(len(points) - 1)]
    gaps = [gap for gap in gaps if gap > 0]
    if not gaps:
        return points
    # Candidate base spacings. The minimum adjacent gap is the safest
    # estimate of the true tick step: when a tick is missed the gap becomes
    # an exact multiple, so the median is biased toward the doubled spacing
    # (e.g. [134,134,66,67] has median 134 although the true step is ~67).
    # Also consider fractions of larger gaps so a stray spurious detection
    # cannot shrink the step below the dominant interval.
    min_gap = min(gaps)
    candidates = {min_gap}
    for gap in gaps:
        for factor in (2, 3, 4):
            step = gap / factor
            if step >= 8:
                candidates.add(step)
    best: list[float] | None = None
    best_score = -1
    for spacing in sorted(candidates):
        for start in points:
            grid = [start + step * spacing for step in range(count)]
            score = sum(
                1
                for value in points
                if any(abs(value - position) < tolerance for position in grid)
            )
            if score > best_score or (score == best_score and spacing < (best[1] if best else float("inf"))):
                best_score = score
                best = (grid, spacing)
    if best is None or best_score < 2:
        return points
    return best[0]

def _thin_binary(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning: binary 0/255 -> single-pixel skeleton."""
    skeleton = binary.copy()
    height, width = skeleton.shape
    changed = True
    while changed:
        changed = False
        for pass_index in (1, 2):
            markers: list[tuple[int, int]] = []
            for y in range(1, height - 1):
                row = skeleton[y]
                row_prev = skeleton[y - 1]
                row_next = skeleton[y + 1]
                for x in range(1, width - 1):
                    if row[x] != 255:
                        continue
                    p2 = row_prev[x] // 255
                    p3 = row_prev[x + 1] // 255
                    p4 = row[x + 1] // 255
                    p5 = row_next[x + 1] // 255
                    p6 = row_next[x] // 255
                    p7 = row_next[x - 1] // 255
                    p8 = row[x - 1] // 255
                    p9 = row_prev[x - 1] // 255
                    neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
                    transitions = sum(
                        1
                        for index in range(8)
                        if neighbors[index] == 0 and neighbors[(index + 1) % 8] == 1
                    )
                    total = sum(neighbors)
                    if not (2 <= total <= 6 and transitions == 1):
                        continue
                    if pass_index == 1:
                        if p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0:
                            markers.append((x, y))
                    else:
                        if p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0:
                            markers.append((x, y))
            for x, y in markers:
                skeleton[y, x] = 0
            if markers:
                changed = True
    return skeleton

def _line_trace_yx(
    skeleton: np.ndarray,
    x_start: int,
    x_end: int,
    max_step: float = 30.0,
) -> list[tuple[float, float]]:
    """Sample the skeleton column-by-column into an ordered (x, y) polyline.

    A thin polyline has at most a handful of skeleton pixels per column, and
    columns where the polyline crosses a grid line (or a shaded band edge
    merged by dilation) contain pixels from both. The median y stays on the
    line at those crossings; a pure nearest-neighbour trace would jump onto
    the grid line and follow it. At the tail, once the polyline has ended and
    only grid-line remnants remain, the median jumps by more than
    ``max_step`` px and tracing stops, so the tail never extends onto
    non-line pixels. Columns with long vertical runs (text, the y axis) are
    skipped.
    """
    height = skeleton.shape[0]
    points: list[tuple[float, float]] = []
    current_y: float | None = None
    for x in range(x_start, x_end):
        column = np.where(skeleton[:, x] > 0)[0]
        if column.size == 0:
            continue
        if column.size > 14:
            continue
        ys = column.astype(float)
        median_y = float(np.median(ys))
        if current_y is None:
            pick = median_y
        elif abs(median_y - current_y) <= max_step:
            pick = median_y
        else:
            # The median was dragged away from the line (tail remnants, a
            # thick crossing): only continue if a candidate still sits near
            # the previous y; otherwise the polyline has ended.
            nearest = float(min(ys, key=lambda value: abs(value - current_y)))
            if abs(nearest - current_y) > max_step:
                break
            pick = nearest
        points.append((float(x), pick))
        current_y = pick
    return points


def _remove_horizontal_lines(
    skeleton: np.ndarray,
    min_share: float = 0.5,
    min_span: float = 0.9,
) -> np.ndarray:
    """Remove horizontal grid-line remnants from a thinned component.

    Dilation merges grid lines with the polyline, and every column then
    contains the grid line's fixed y. Such rows appear in most columns and
    would drag a global path trace onto the flat line (the perfect smooth
    path). Rows covered by >= ``min_share`` of the columns are treated as
    horizontal lines and removed. A polyline's horizontal plateau (e.g. a
    wage line that flattens out) also repeats on a fixed row across many
    columns, but it only spans part of the component width: it starts/ends at
    a turn and does not run edge-to-edge. Requiring the row to cover >=
    ``min_span`` of the width keeps plateaus while still removing true
    edge-to-edge grid lines.
    """
    height, width = skeleton.shape
    rows = np.where(skeleton > 0)[0]
    if rows.size == 0:
        return skeleton
    column_counts = np.bincount(rows, minlength=height)
    removable = np.zeros(height, dtype=bool)
    for row in range(height):
        if column_counts[row] < width * min_share:
            continue
        cols = np.where(skeleton[row] > 0)[0]
        if cols.size == 0:
            continue
        span = (int(cols.max()) - int(cols.min()) + 1) / width
        if span >= min_span:
            removable[row] = True
    if not np.any(removable):
        return skeleton
    cleaned = skeleton.copy()
    for row in np.where(removable)[0]:
        cleaned[row, :] = 0
    return cleaned


def _trace_dp_path(
    skeleton: np.ndarray,
    *,
    max_gap: int = 3,
    max_step: float = 45.0,
) -> list[tuple[float, float]]:
    """Trace the polyline through a noisy component with a global path search.

    Each column contributes its skeleton pixels as candidates; a dynamic
    program picks one candidate per column minimising the total vertical jump
    (plus a fixed penalty per skipped column). Unlike a greedy nearest-column
    trace this tolerates an illustration or text remnant sharing the
    component: the polyline wins because it spans almost every column while
    local decorations only cover a few. ``max_step`` turns jumps above the
    threshold into a large (not infinite) penalty so a single outlier cannot
    derail the whole path.
    """
    cols: list[np.ndarray | None] = []
    for x in range(skeleton.shape[1]):
        ys = np.where(skeleton[:, x] > 0)[0]
        if ys.size == 0 or ys.size > 14:
            cols.append(None)
        else:
            cols.append(ys.astype(float))
    n = len(cols)
    if n == 0:
        return []
    big = 1e9
    dp: list[np.ndarray | None] = [None] * n
    back: list[np.ndarray | None] = [None] * n
    prev_nonempty = -1
    for x in range(n):
        ys = cols[x]
        if ys is None:
            continue
        if prev_nonempty < 0:
            dp[x] = np.zeros(len(ys))
        else:
            gap = x - prev_nonempty
            if gap > max_gap:
                dp[x] = np.zeros(len(ys))
            else:
                diff = np.abs(ys[:, None] - cols[prev_nonempty][None, :])
                cost = np.minimum(diff, max_step + 30.0)
                if gap > 1:
                    cost += 80.0 * (gap - 1)
                idx = cost.argmin(axis=1)
                dp[x] = cost[np.arange(len(ys)), idx]
                back[x] = idx
        prev_nonempty = x

    last = next((x for x in range(n - 1, -1, -1) if cols[x] is not None), None)
    if last is None:
        return []
    j = int(np.argmin(dp[last]))
    points: list[tuple[float, float]] = []
    x = last
    while x >= 0:
        ys = cols[x]
        if ys is not None:
            points.append((float(x), float(ys[j])))
        prev = x - 1
        while prev >= 0 and cols[prev] is None:
            prev -= 1
        if prev < 0:
            break
        if back[x] is not None:
            j = int(back[x][j])
        else:
            # The path restarted at this column; restart at the previous
            # column's cheapest candidate instead.
            j = int(np.argmin(dp[prev]))
        x = prev
    return list(reversed(points))


def _smooth_trace(
    points: list[tuple[float, float]],
    window: int = 5,
) -> list[tuple[float, float]]:
    """Median-filter the y coordinates of a traced polyline to remove
    skeleton jitter (thick lines produce noisy single-pixel skeletons)."""
    if len(points) < window:
        return points
    ys = [point[1] for point in points]
    half = window // 2
    smoothed = []
    for index in range(len(ys)):
        low = max(0, index - half)
        high = min(len(ys), index + half + 1)
        smoothed.append((points[index][0], float(np.median(ys[low:high]))))
    return smoothed


def _trim_path_ends(
    points: list[tuple[float, float]],
    min_dy: float = 150.0,
    min_slope: float = 3.0,
) -> list[tuple[float, float]]:
    """Drop leading/trailing points that jump far away from the path body.

    The global path search starts at the leftmost column of the component,
    which may be a stray element above the real polyline (e.g. a light edge
    of a fill area in the achromatic mask). Such a point attaches with a
    nearly vertical jump (large ``dy`` over a tiny ``dx``), whereas a genuine
    steep line segment still spans many pixels horizontally, so the slope
    keeps real endpoints while dropping stray ones.
    """
    points = list(points)
    while len(points) >= 2:
        (x0, y0), (x1, y1) = points[0], points[1]
        dy = abs(y1 - y0)
        if dy > min_dy and dy > min_slope * max(abs(x1 - x0), 1.0):
            points.pop(0)
        else:
            break
    while len(points) >= 2:
        (x0, y0), (x1, y1) = points[-2], points[-1]
        dy = abs(y1 - y0)
        if dy > min_dy and dy > min_slope * max(abs(x1 - x0), 1.0):
            points.pop()
        else:
            break
    return points


def _simplify_polyline(
    points: list[tuple[float, float]],
    epsilon: float = 5.0,
) -> list[tuple[float, float]]:
    """Ramer-Douglas-Peucker simplification of a polyline."""
    if len(points) <= 2:
        return points

    def _distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
        sx, sy = start
        ex, ey = end
        px, py = point
        dx, dy = ex - sx, ey - sy
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return float(((px - sx) ** 2 + (py - sy) ** 2) ** 0.5)
        t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / length_sq))
        proj_x, proj_y = sx + t * dx, sy + t * dy
        return float(((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5)

    stack = [(0, len(points) - 1)]
    keep = {0, len(points) - 1}
    while stack:
        start, end = stack.pop()
        if end - start < 2:
            continue
        max_dist = 0.0
        max_index = start
        for index in range(start + 1, end):
            dist = _distance(points[index], points[start], points[end])
            if dist > max_dist:
                max_dist = dist
                max_index = index
        if max_dist > epsilon:
            keep.add(max_index)
            stack.append((start, max_index))
            stack.append((max_index, end))
    return [points[index] for index in sorted(keep)]


def _line_color_masks(img: np.ndarray) -> list[np.ndarray]:
    """Cluster the image into per-colour masks, one per dominant line colour.

    Multi-series charts draw each polyline in a different colour; a single
    Canny pass merges lines whose colours are close and lets one line absorb
    the other. Coloured lines are clustered by hue peaks; achromatic lines
    (white/grey/black) are handled by a brightness layer relative to the
    median background. Fill areas / illustrations that share a hue with a
    line are handled later by the trace + variance filters.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.int16)
    hue = hsv[:, :, 0].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    background = float(np.median(gray))

    masks: list[np.ndarray] = []
    # Achromatic lines (white/grey/black) and lightly tinted greys often sit
    # right at a fixed saturation cutoff; use a wide <=70 band for the
    # brightness layers and a >45 band for hue clusters so a light-grey line
    # never gets sliced into fragments.
    colored = saturation > 45
    if int(colored.sum()) > 200:
        histogram = np.bincount(hue[colored], minlength=180).astype(float)
        histogram = np.convolve(histogram, np.ones(3) / 3.0, mode="same")
        threshold = max(100.0, 0.004 * int(colored.sum()))
        peaks = [
            h
            for h in range(180)
            if histogram[h] >= threshold
            and histogram[h] >= histogram[(h - 1) % 180]
            and histogram[h] >= histogram[(h + 1) % 180]
        ]
        merged: list[int] = []
        for peak in peaks:
            if merged and abs(peak - merged[-1]) <= 8:
                continue
            merged.append(peak)
        for peak in merged:
            distance = np.abs(hue - peak)
            distance = np.minimum(distance, 180 - distance)
            masks.append(((distance <= 12) & (saturation > 45)).astype(np.uint8) * 255)

    if background < 128:
        masks.append(((saturation <= 70) & (value > background + 50)).astype(np.uint8) * 255)
    else:
        masks.append(((saturation <= 70) & (value < background - 50)).astype(np.uint8) * 255)
    return masks


def detect_lines(
    image_path: str | Path,
) -> list[dict[str, Any]]:
    """Detect line-chart polylines and return their data points.

    Returns a list of ``{"points": [(x, y), ...], "color": (b, g, r)}`` where
    ``points`` are the pixel coordinates of the polyline vertices (data
    points). The image is first clustered by colour so multi-series charts
    yield one polyline per series; each colour mask is then thinned to a
    single-pixel skeleton, traced with the global path search and simplified
    with RDP. Grid-line remnants are removed before tracing.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    plot_w = int(width * 0.8)
    y_min = int(height * 0.15)
    x_ticks = _detect_x_axis_tick_positions(img)
    lines: list[dict[str, Any]] = []
    seen_bboxes: list[tuple[int, int, int, int]] = []

    for mask in _line_color_masks(img):
        # Close horizontal gaps (line pixels flicker across the saturation
        # cutoff), then dilate so nearby strokes merge into one component.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
        dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(dilated, 8)
        for i in range(1, n):
            x, y, w, h, area = [int(v) for v in stats[i]]
            # Illustrations or textured backgrounds often merge with the
            # polyline into one tall component (e.g. an IKEA chair next to the
            # line); keep those candidates and let the trace + variance
            # filters reject non-line content.
            if w < plot_w * 0.35 or h < 16 or h > height * 0.75 or y < y_min:
                continue
            if area < w * 1.5 or area > w * 200:
                continue
            component_mask = (labels == i).astype(np.uint8) * 255
            cropped = component_mask[y : y + h, x : x + w]
            skeleton = _thin_binary(cropped)
            skeleton = _remove_horizontal_lines(skeleton)
            traced = _trace_dp_path(skeleton)
            if len(traced) < 5:
                traced = _line_trace_yx(skeleton, 0, w)
            if len(traced) < 5:
                continue
            traced = [(px + x, py + y) for px, py in traced]
            traced = _smooth_trace(traced)
            # A real polyline yields a skeleton column on nearly every x
            # inside its bounding box; stray text/decoration remnants cover
            # only a few columns even though their dilated component spans
            # the width.
            if len(traced) < int(w * 0.5):
                continue
            # A polyline has real vertical variation; near-horizontal long
            # components (a stray grid line remnant, a title underline, a
            # decoration) are not data lines and must be rejected generically.
            trace_ys = [py for _, py in traced]
            if max(trace_ys) - min(trace_ys) < 40 or float(np.std(trace_ys)) < 20.0:
                continue
            extent = max(w, h)
            epsilon = max(8.0, extent * 0.02)
            vertices = _simplify_polyline(traced, epsilon=epsilon)
            if len(vertices) < 3:
                # Smooth curve: align to x-axis ticks (or sample evenly).
                vertices = _curve_data_points(traced, x_ticks)
            vertices = _trim_path_ends(vertices)
            if len(vertices) < 2:
                continue
            # Data-line validation: a real polyline spans most of the plot
            # width (the time axis runs left-to-right) and starts in the left
            # part of the frame. Arrows, annotations and broken fragments are
            # local strokes that cover only a fraction of the width; they
            # must not be emitted as data series.
            xs = [float(px) for px, _ in vertices]
            xspan = (max(xs) - min(xs)) / width
            left_ratio = min(xs) / width
            right_ratio = max(xs) / width
            if xspan < 0.35 or left_ratio >= 0.45 or right_ratio < 0.55:
                continue
            # Skip near-duplicate lines (the same stroke can appear in two
            # hue clusters at the cluster boundary).
            bbox = (x, y, x + w, y + h)
            if any(_bbox_overlap(bbox, seen) >= 0.75 for seen in seen_bboxes):
                continue
            seen_bboxes.append(bbox)
            color = img[component_mask > 0].mean(axis=0)
            lines.append(
                {
                    "points": [(round(float(px)), round(float(py))) for px, py in vertices],
                    "color": [int(v) for v in color],
                    "bbox": [x, y, x + w, y + h],
                }
            )
    return lines


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU of two bounding boxes (used to dedupe lines across hue clusters)."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _detect_x_axis_tick_positions(img: np.ndarray) -> list[float]:
    """Detect the x-axis tick positions (short vertical strokes above the
    bottom axis), used to sample smooth line charts at the plotted years."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    background = float(np.median(gray))
    mask = np.abs(gray.astype(np.int16) - background) > 25
    binary = mask.astype(np.uint8) * 255
    y_start = int(height * 0.88)
    positions: list[float] = []
    for x in range(int(width * 0.08), int(width * 0.97)):
        column = np.where(binary[y_start:, x] > 0)[0]
        if column.size == 0:
            continue
        if column.size > 14:
            continue
        positions.append(float(x))
    merged: list[float] = []
    for position in positions:
        if not merged or position - merged[-1] > 6.0:
            merged.append(position)
    return merged


def _curve_data_points(
    traced: list[tuple[float, float]],
    x_ticks: list[float],
    max_points: int = 10,
) -> list[tuple[float, float]]:
    """Sample a smooth curve at the x-axis tick positions (fallback: even
    sampling) so curved charts still yield meaningful data points."""
    if not traced:
        return []
    if len(x_ticks) >= 3:
        points = []
        for tick in x_ticks:
            nearest = min(traced, key=lambda point: abs(point[0] - tick))
            points.append(nearest)
        return points
    step = max(1, len(traced) // max_points)
    return [traced[index] for index in range(0, len(traced), step)]


def _cluster_consecutive(values: list[int], *, gap: int) -> list[list[int]]:
    """Group sorted ints into consecutive bands (gaps <= ``gap`` are merged)."""
    bands: list[list[int]] = []
    for value in sorted(values):
        if bands and value - bands[-1][-1] <= gap:
            bands[-1].append(value)
        else:
            bands.append([value])
    return bands

def _parse_tick_labels(text: str) -> tuple[list[float], str]:
    """Parse a vision JSON array of axis tick labels into numbers + unit."""
    unit = "$" if "$" in text else ("%" if "%" in text else "")
    suffix_unit = ""
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return [], unit
    try:
        raw = json.loads(match.group(0))
    except Exception:
        return [], unit
    if not isinstance(raw, list):
        return [], unit
    labels: list[float] = []
    for entry in raw:
        text_entry = str(entry).strip()
        low_entry = text_entry.lower()
        if low_entry.endswith("k"):
            suffix_unit = "k"
        elif low_entry.endswith("m"):
            suffix_unit = "m"
        elif low_entry.endswith("b"):
            suffix_unit = "b"
        digits = re.sub(r"[^0-9.\-]", "", text_entry)
        if not digits:
            return [], unit
        try:
            # Preserve the chart's own scaling: "20k" stays 20 with unit "k",
            # "20000" stays 20000 with no suffix.  Do not silently rescale.
            labels.append(float(digits))
        except ValueError:
            return [], unit
    if suffix_unit:
        unit = unit + suffix_unit if unit else suffix_unit
    return labels, unit


def read_tick_labels(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
    orientation: str = "vertical",
) -> tuple[list[float], str]:
    """Read the value-axis tick labels via three full-frame vision calls.

    Returns ``(labels, unit)`` where labels are ordered along the value axis
    (bottom-to-top for vertical bars, left-to-right for horizontal bars).
    Three attempts are majority-voted because the vision model occasionally
    misreads magnitudes (e.g. 15,000 / 30,000 as 8,000 / 16,000), which would
    silently halve the whole value scale.
    """
    if orientation == "horizontal":
        prompt = (
            "这是横向条形图。请读出底部横轴（数值轴）上的刻度标签，"
            "从左到右依次列出，只返回 JSON 数字数组，例如 [0, 100, 200]。"
            '若标签带 "$" "%" "k" "M" 等符号请保留原样，例如 ["$0", "$50k", "$100k"]。'
            "必须精确读出每个数字的完整大小（如 15,000 与 30,000 不能读成 8,000/16,000），不要解释。"
        )
    else:
        prompt = (
            "这是柱状图。请读出左侧纵轴（数值轴）上的刻度标签，"
            "从下到上依次列出，只返回 JSON 数字数组，例如 [0, 10, 20]。"
            '若标签带 "$" "%" "k" "M" 等符号请保留原样，例如 ["0%", "50k", "100k"]。'
            "必须精确读出每个数字的完整大小（如 15,000 与 30,000 不能读成 8,000/16,000），不要解释。"
        )
    attempts: list[tuple[list[float], str]] = []
    for _ in range(3):
        try:
            text = _call_vision(image_path, prompt, cfg, temperature=0.0)
        except Exception:
            continue
        labels, unit = _parse_tick_labels(text)
        if labels:
            attempts.append((labels, unit))
    if not attempts:
        return [], ""
    # Majority vote: identical label sequences win; otherwise prefer the
    # sequence with the largest values (under-reading magnitudes is the
    # failure mode, e.g. 30,000 misread as 16,000).
    best = attempts[0]
    best_count = 1
    for index in range(len(attempts)):
        count = sum(1 for other in attempts if other[0] == attempts[index][0])
        if count > best_count:
            best = attempts[index]
            best_count = count
    if best_count == 1:
        best = max(attempts, key=lambda item: sum(item[0]))
    return best


def _pair_ticks_with_labels(
    tick_marks: list[dict[str, Any]],
    labels: list[float],
    orientation: str = "vertical",
) -> list[dict[str, Any]]:
    """Pair detected tick coords with vision-read labels in value order."""
    if not tick_marks or len(labels) != len(tick_marks):
        return []
    reverse = orientation != "horizontal"
    ordered = sorted(tick_marks, key=lambda item: float(item["coord"]), reverse=reverse)
    return [
        {"coord": float(item["coord"]), "value": float(value)}
        for item, value in zip(ordered, labels)
    ]


def read_x_axis_labels(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Read the x-axis (category) labels printed under a vertical chart.

    Returns the labels left-to-right as strings (e.g. ``["2000-01", "2010-11",
    "2018-09"]``). The value axis labels are read by ``read_tick_labels``.
    """
    prompt = (
        "这是一张图表。请读出底部横轴（类别轴）上的刻度标签，从左到右依次列出，"
        '只返回 JSON 字符串数组，例如 ["2000", "2010", "2020"]；'
        "不要包含纵轴标签、图例或标题，不要解释。"
    )
    try:
        text = _call_vision(image_path, prompt, cfg, temperature=0.0)
    except Exception:
        return []
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def read_series_label(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> str:
    """Read the metric/series name a line chart's polyline represents.

    Single-series line charts usually print the series name in a legend or in
    the title ("'Net additions' in England"); asking the vision model directly
    is more reliable than deriving it from the title with string heuristics.
    """
    prompt = (
        "这是一张折线图。读出图中折线所代表的指标/系列名称：优先读图例或标题中"
        "明确写出的名称；如果图上没有明确写出，根据图表内容和上下文推断一个简短"
        "准确的名称。不要使用坐标轴单位说明（如 2016 Dollars）或目标区标注作为名称。"
        "只返回名称本身，不要解释。"
    )
    try:
        text = _call_vision(image_path, prompt, cfg, temperature=0.0)
    except Exception:
        return ""
    title = text.strip().strip("\"'`。，,.")
    if title.upper() in {"NONE", "无", "没有", "无系列名"}:
        return ""
    return title


def read_series_labels(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> list[str]:
    """Read all series names from a multi-series line chart's legend.

    Returns names in the same visual order as the polylines (top-to-bottom
    when the legend is vertical). Falls back to a single-label read so
    single-series charts keep working.
    """
    prompt = (
        "这是一张折线图，可能有多条折线。请只读出图例中所有系列的名称，"
        "按视觉顺序（从上到下/从左到右）列出，"
        '只返回 JSON 字符串数组，例如 ["Productivity", "Wages"]。'
        "如果图上没有图例（系列名只出现在标题或坐标轴说明里），返回空数组 []，"
        "不要从标题、目标区标注或图例说明中猜测。"
    )
    try:
        text = _call_vision(image_path, prompt, cfg, temperature=0.0)
    except Exception:
        return []
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    labels = [str(item).strip() for item in raw if str(item).strip()]
    # A unit / annotation caption (e.g. "2016 Dollars", "$350") is not a
    # series name; dropping it lets the pipeline fall back to the metric or
    # title-derived label instead of inventing a legend entry.
    labels = [
        label
        for label in labels
        if not re.search(r"(^\d{4}\s+\w+$)|(dollars?$)|(usd$)|(^\$\d)", label, re.I)
    ]
    # A real legend entry is a short series name; long concatenated text
    # (e.g. the vision model merging the title with a target-zone annotation)
    # is not a legend entry and must not be used.
    expanded: list[str] = []
    for label in labels:
        if "," in label:
            expanded.extend(part.strip() for part in label.split(",") if part.strip())
        else:
            expanded.append(label)
    labels = [label for label in expanded if len(label) <= 40]
    if not labels:
        single = read_series_label(image_path, cfg)
        if single:
            labels = [single]
    return labels


def read_line_data(
    image_path: str | Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read each series' key turning points via the vision model.

    CV tracing can be misled by anti-aliased ghost strokes next to a line
    (especially on steep multi-series charts), so the printed values read by
    the vision model are the authoritative source. Returns
    ``{"unit", "series": [{"name", "points": [[year, value], ...]}]}`` with
    the start, end and peak/trough turning points, merged over three vision
    calls (nearby years clustered, values medianed).
    """
    prompt = (
        "这是一张折线图（可能有多条线），横轴是刻度（可能是年份、月份或其他标签）。"
        "请为每条折线读出10到15个数据点，从横轴起点到最右端终点均匀分布，"
        "必须包含起点和终点，以及所有明显的峰/谷转折点和途中均匀的中间点。"
        "不要让后半段缺失。每个点给出（横轴标签, 数值）。"
        "只返回JSON：{\"unit\":\"图中y轴刻度使用的单位符号（如$、%、k、M；"
        "若图中没有任何单位符号则填空字符串\\\"\\\"，禁止臆测单位）\","
        "\"series\":[{\"name\":\"系列名\","
        "\"points\":[[标签,数值],[标签,数值],...]}]}。按横轴顺序排列，"
        "点要均匀覆盖整条线，从起点一直读到终点，不要省略后半段。"
        "不要计算，按线在y轴刻度处的位置读数。"
    )
    attempts: list[tuple[dict[str, list[tuple[float, float]]], str]] = []
    for _ in range(3):
        try:
            text = _call_vision(image_path, prompt, cfg, temperature=0.0)
        except Exception:
            continue
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            continue
        try:
            raw = json.loads(match.group(0))
        except Exception:
            continue
        if not isinstance(raw, dict):
            continue
        series: dict[str, list[tuple[str, float]]] = {}
        for item in raw.get("series", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            points = item.get("points")
            if not name or not isinstance(points, list):
                continue
            parsed: list[tuple[str, float]] = []
            for point in points:
                try:
                    label = str(point[0]).strip()
                    value = float(point[1])
                except (TypeError, ValueError, IndexError):
                    continue
                parsed.append((label, value))
            if parsed:
                series[name] = parsed
        if series:
            attempts.append((series, str(raw.get("unit") or "").strip()))
    if not attempts:
        return {}

    def _numeric(label: str) -> float | None:
        try:
            return float(label)
        except (TypeError, ValueError):
            return None

    # The vision responses differ in which turning points they report, so a
    # per-point median merge dilutes a good dense reading down to the
    # intersection of the sparse ones. Prefer the densest response and keep
    # its points as-is (all series must come from the same reading so their
    # x labels stay aligned).
    best = max(
        attempts,
        key=lambda attempt: sum(len(points) for points in attempt[0].values()),
    )
    best_series, unit = best
    merged = [
        {"name": name, "points": points}
        for name, points in best_series.items()
        if points
    ]
    if not merged:
        return {}
    return {
        "unit": unit,
        "series": merged,
    }


def _assign_x_labels(
    points: list[dict[str, Any]],
    x_ticks: list[float],
    labels: list[str],
    left: float,
    right: float,
) -> None:
    """Attach an x-axis label to each data point.

    Year-like labels (e.g. ``2000-01``) are interpolated continuously so each
    point gets its own year instead of snapping to the nearest printed tick;
    non-year category labels (Jan/Feb, A/B/C) snap to the nearest label.
    """
    if not labels:
        return
    if len(labels) == 1:
        for point in points:
            point["x_label"] = labels[0]
        return
    if len(x_ticks) >= 2:
        anchor0 = float(x_ticks[0])
        anchor1 = float(x_ticks[-1])
    else:
        anchor0 = float(left)
        anchor1 = float(right)
    span = anchor1 - anchor0
    if span <= 0:
        return
    year_values = [_year_value(label) for label in labels]
    year_like = all(value is not None for value in year_values) and len(year_values) >= 2
    for point in points:
        px = float(point.get("x") or 0.0)
        ratio = (px - anchor0) / span
        if year_like:
            year = year_values[0] + ratio * (year_values[-1] - year_values[0])
            point["x_label"] = _format_year_label(year, labels[0])
        else:
            index = ratio * (len(labels) - 1)
            point["x_label"] = labels[min(len(labels) - 1, max(0, round(index)))]


def _year_value(label: Any) -> float | None:
    match = re.search(r"(?:19|20)\d{2}", str(label or ""))
    return float(match.group(0)) if match else None


def _format_year_label(year: float, template: str) -> str:
    """Format an interpolated year using the printed label's style.

    ``2000-01`` (fiscal years) -> ``2007-08``; plain ``2000`` -> ``2007``.
    """
    rounded = int(round(year))
    if "-" in str(template):
        return f"{rounded}-{str(rounded + 1)[-2:]}"
    return str(rounded)


def _infer_baseline_coord(
    aligned: list[dict[str, Any]],
    orientation: str = "vertical",
) -> float | None:
    """Infer the value-axis baseline (the 0/start anchor) from bar geometry.

    Vertical bars share a bottom edge; horizontal bars share a left edge
    (or right edge for right-aligned charts, handled by orientation of the
    majority). This anchor is usually missing from grid-line detection
    because the bars' own pixels merge with the baseline.
    """
    if orientation == "horizontal":
        xs = [
            float(item["x"])
            for item in aligned
            if item.get("x") is not None and _bar_length(item) > 0
        ]
        return min(xs) if xs else None
    bottoms = [
        float(item["y"]) + float(item["h"])
        for item in aligned
        if item.get("y") is not None and item.get("h") is not None and float(item["h"]) > 0
    ]
    return max(bottoms) if bottoms else None


def _tick_scale(tick_marks: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Build a sorted (coords, values, slope) linear scale from tick marks."""
    ticks = sorted(
        (
            (float(item["coord"]), float(item["value"]))
            for item in tick_marks
            if item.get("coord") is not None and item.get("value") is not None
        ),
        key=lambda pair: pair[0],
    )
    if len(ticks) < 2:
        return None
    coords = np.array([pair[0] for pair in ticks], dtype=float)
    values = np.array([pair[1] for pair in ticks], dtype=float)
    if np.ptp(coords) == 0 or np.ptp(values) == 0 or not np.all(np.isfinite(values)):
        return None
    slope = (values[-1] - values[0]) / (coords[-1] - coords[0])
    return coords, values, slope


def _estimate_coord_value(coord: float, scale: tuple[np.ndarray, np.ndarray, float]) -> float:
    """Map an axis coordinate to a value (interpolate inside, extrapolate
    beyond the drawn ticks)."""
    coords, values, slope = scale
    if coord <= coords[0]:
        estimated = values[0] + slope * (coord - coords[0])
    elif coord >= coords[-1]:
        estimated = values[-1] + slope * (coord - coords[-1])
    else:
        estimated = float(np.interp(coord, coords, values))
    return max(0.0, estimated)


def bar_layout_regularity(bars: list[dict[str, Any]]) -> float:
    """Score how regularly bars are laid out along the category axis (0..1).

    A clean chart has evenly spaced, same-width bars. A cross-fade frame
    (two charts superimposed, e.g. a Vox transition) shows duplicated and
    misaligned bars: irregular gaps along the category axis and inconsistent
    widths for vertical charts. Such frames must never become keyframes
    because their tick marks and bars come from two different charts.
    """
    if len(bars) < 3:
        return 1.0
    first = bars[0]
    orientation = str(first.get("orientation") or "")
    if not orientation:
        orientation = "horizontal" if float(first.get("w") or 0.0) >= float(first.get("h") or 0.0) else "vertical"
    if orientation == "horizontal":
        positions = sorted(float(bar.get("y") or 0.0) for bar in bars)
        widths = [float(bar.get("w") or 0.0) for bar in bars]
    else:
        positions = sorted(float(bar.get("x") or 0.0) for bar in bars)
        widths = [float(bar.get("w") or 0.0) for bar in bars]
    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    if not gaps or max(gaps) <= 0:
        return 1.0
    mean_gap = sum(gaps) / len(gaps)
    gap_cv = (sum((gap - mean_gap) ** 2 for gap in gaps) / len(gaps)) ** 0.5 / mean_gap
    mean_width = sum(widths) / len(widths) if widths else 0.0
    width_cv = (
        (sum((width - mean_width) ** 2 for width in widths) / len(widths)) ** 0.5 / mean_width
        if mean_width
        else 1.0
    )
    if orientation == "horizontal":
        # Width encodes the value for horizontal bars; only spacing matters.
        return max(0.0, 1.0 - gap_cv * 3.0)
    return max(0.0, 1.0 - (gap_cv + width_cv) * 2.0)


def estimate_unlabeled_values_from_ticks(
    aligned: list[dict[str, Any]],
    tick_marks: list[dict[str, Any]],
) -> int:
    """Estimate unlabeled bar values from value-axis tick marks.

    ``tick_marks`` are ``{"coord", "value"}`` pairs along the value axis (y
    for vertical bars, x for horizontal bars). Values are linearly
    interpolated between adjacent ticks; bars that extend beyond the drawn
    axis (e.g. the US bar in the drug-price charts) are extrapolated along the
    tick scale instead of being clamped to the last tick value.
    """
    scale = _tick_scale(tick_marks)
    if scale is None:
        return 0
    count = 0
    for item in aligned:
        if item.get("value") is not None:
            continue
        length = _bar_length(item)
        if length <= 0:
            continue
        if str(item.get("orientation") or "") == "horizontal":
            coord = float(item.get("x") or 0.0) + length
        else:
            coord = float(item.get("y") or 0.0)
        if not np.isfinite(coord):
            continue
        estimated = _estimate_coord_value(coord, scale)
        item["value"] = estimated
        item["value_text"] = f"{estimated:.0f}"
        item["value_estimated"] = True
        item["value_type"] = "estimated"
        item["plausibility_message"] = "estimated from axis tick scale"
        item["value_read_verified"] = False
        count += 1
    return count


def run_cv_align_line(
    clip_id: str,
    image_path: str | Path,
    out_dir: str | Path,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Line-chart CV alignment: detect polylines, read the axis tick scale
    and estimate each data point's value from its y coordinate."""
    out_dir = ensure_dir(out_dir)
    lines = detect_lines(image_path)
    x_ticks: list[float] = []
    x_labels: list[str] = []
    try:
        frame = cv2.imread(str(image_path))
        if frame is not None:
            x_ticks = _detect_x_axis_tick_positions(frame)
        x_labels = read_x_axis_labels(image_path, cfg)
    except Exception:
        pass
    tick_marks: list[dict[str, Any]] = []
    tick_unit = ""
    try:
        tick_marks = detect_axis_tick_marks(image_path, "vertical")
    except Exception:
        tick_marks = []
    if tick_marks:
        try:
            tick_labels, tick_unit = read_tick_labels(image_path, cfg, "vertical")
        except Exception:
            tick_labels = []
        paired = _pair_ticks_with_labels(tick_marks, tick_labels, "vertical") if tick_labels else []
        if not paired and tick_labels and len(tick_labels) != len(tick_marks):
            # CV tick detections can latch onto unrelated horizontal lines
            # (axis baselines, decorations) when the chart has no grid lines.
            # The printed tick labels are the most reliable anchors then:
            # rebuild an even set over their text-block positions.
            frame = cv2.imread(str(image_path))
            height = frame.shape[0] if frame is not None else 720
            label_blocks = (
                _detect_tick_label_blocks(frame, "vertical") if frame is not None else []
            )
            cv_coords = [float(item["coord"]) for item in tick_marks]
            # CV grid-line / stroke detections are the preferred anchor:
            # they cover the full axis (including labels the vision block
            # scan may miss at the top/bottom). Label blocks are only used
            # when the CV ticks are degenerate, e.g. two bottom decorations
            # latched instead of the real axis.
            if cv_coords and (max(cv_coords) - min(cv_coords)) >= 0.2 * height:
                base = cv_coords
            else:
                base = label_blocks if len(label_blocks) >= 2 else cv_coords
            rebuilt = _rebuild_even_ticks(
                base,
                len(tick_labels),
            )
            paired = _pair_ticks_with_labels(
                [{"coord": coord} for coord in rebuilt],
                tick_labels,
                "vertical",
            )
        tick_marks = paired
    scale = _tick_scale(tick_marks) if tick_marks else None

    point_count = 0
    series_labels: list[str] = []
    try:
        series_labels = read_series_labels(image_path, cfg)
    except Exception:
        series_labels = []
    for line in lines:
        enriched_points = []
        bbox = [float(value) for value in line.get("bbox", [0, 0, 1, 1])]
        left, right = bbox[0], bbox[2]
        for px, py in line.get("points", []):
            value = _estimate_coord_value(float(py), scale) if scale is not None else None
            enriched_points.append({"x": int(px), "y": int(py), "value": value})
            if value is not None:
                point_count += 1
        _assign_x_labels(enriched_points, x_ticks, x_labels, left, right)
        line["points"] = enriched_points
    for index, line in enumerate(lines):
        if series_labels:
            line["label"] = (
                series_labels[index] if index < len(series_labels) else f"series {index + 1}"
            )
    vision_data: dict[str, Any] = {}
    try:
        vision_data = read_line_data(image_path, cfg)
    except Exception:
        vision_data = {}
    if vision_data.get("series"):
        # The vision model reads each series' value at every x-axis tick. CV
        # tracing can be misled by anti-aliased ghost strokes next to a line,
        # so the printed values are the authoritative source; match vision
        # series to detected lines by label first, then by order.
        used: set[int] = set()
        for line in lines:
            label = str(line.get("label") or "")
            match = next(
                (s for s in vision_data["series"] if s["name"] == label and id(s) not in used),
                None,
            )
            if match is None:
                match = next((s for s in vision_data["series"] if id(s) not in used), None)
            if match is None:
                continue
            used.add(id(match))
            # The vision data carries the accurate per-series name (the CV
            # legend reading can merge two entries into "A, B"); use it so
            # the rendered legend matches the data.
            line["label"] = match["name"]
            points = match.get("points") or []
            vision_points = [
                {"x_label": str(year), "value": float(value)}
                for year, value in points
            ]
            if len(vision_points) >= 3:
                line["points"] = vision_points
                line["value_source"] = "vision_tick_read"
            else:
                # The vision response was too sparse to replace the CV trace
                # (occasionally the model returns only one point). Keep the
                # CV-estimated polyline as a fallback and flag it for review.
                line["value_source"] = "cv_tick_scale"
                line["value_review"] = True
        if vision_data.get("unit"):
            tick_unit = str(vision_data["unit"])
    report = {
        "clip_id": clip_id,
        "line_count": len(lines),
        "point_count": point_count,
        "tick_mark_count": len(tick_marks),
        "tick_unit": tick_unit,
        "x_axis_tick_count": len(x_ticks),
        "x_axis_labels": x_labels,
        "vision_data": vision_data,
        "value_read_method": "tick_scale" if scale is not None else "none",
        "lines": lines,
        "success": bool(lines),
    }
    write_json(out_dir / "aligned_lines_report.json", report)
    return report


def reconcile_line_dynamic(
    lines: list[dict[str, Any]],
    *,
    clip_id: str,
    image_path: str | Path,
    keyframe_timestamp: float | None,
    unit: str = "",
    series_label: str | None = None,
) -> dict[str, Any]:
    """Turn detected line-chart points into dynamic data rows.

    Every data point becomes one row (series entity + point value); unlike
    bars, a line chart keeps all points in the final table (the value axis
    is continuous, there is no single final state).
    """
    states: list[dict[str, Any]] = []
    for series in lines:
        name = str(series.get("label") or series_label or "series")
        entity_id = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "series"
        for index, point in enumerate(series.get("points", [])):
            value = point.get("value")
            if value is None:
                continue
            # Prefer the x-axis category label (e.g. the year) as the state
            # key so the reconciled intent can say "from 2000 to 2018" instead
            # of "from 0 to 7". Fall back to the point index.
            label = str(point.get("x_label") or index)
            states.append(
                {
                    "clip_id": clip_id,
                    "state_id": f"state_{len(states) + 1:03d}",
                    "state_key": label,
                    "state_label": label,
                    "entity_id": entity_id,
                    "entity": name,
                    "metric": "",
                    "value": value,
                    "unit": unit,
                    "value_type": "estimated",
                    "source_type": "visual_line_estimate",
                    "state_start": keyframe_timestamp,
                    "state_end": keyframe_timestamp,
                    "evidence_frames": [
                        {
                            "frame_id": Path(image_path).stem,
                            "time_seconds": keyframe_timestamp,
                            "path": str(image_path),
                        }
                    ],
                    "confidence": 0.7,
                    "review_status": "machine",
                    "needs_review": True,
                    "raw_text": None,
                    "evidence_text": f"{value:g}",
                }
            )
    return {
        "clip_id": clip_id,
        "states": states,
        "final_data_table": [dict(row) for row in states],
        "data_change_events": [],
        "excluded": not states,
        "exclude_reason": "no_recoverable_quantitative_data" if not states else None,
        "include_in_dataset": bool(states),
        "data_completeness": "complete" if states else "none",
        "numeric_fact_count": len(states),
        "dynamic_data": len(states) > 1,
        "data_change_count": 0,
        "visual_record_count": len(states),
        "narration_record_count": 0,
    }


def _text_width_estimate(text: str, font_size: float) -> float:
    return max(10.0, len(str(text)) * 0.55 * font_size)


def _contrast_outline_color(img: np.ndarray, box: list[int]) -> tuple[int, int, int]:
    """Pick white on dark surroundings and black on light surroundings so the
    text outline stays visible on any chart background."""
    x1, y1, x2, y2 = [int(v) for v in box]
    H, W = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return (255, 255, 255)
    pad = 4
    top = img[max(0, y1 - pad) : y1, max(0, x1 - pad) : min(W, x2 + pad)]
    bottom = img[min(H, y2) : min(H, y2 + pad), max(0, x1 - pad) : min(W, x2 + pad)]
    left = img[max(0, y1 - pad) : min(H, y2 + pad), max(0, x1 - pad) : x1]
    right = img[max(0, y1 - pad) : min(H, y2 + pad), x2 : min(W, x2 + pad)]
    parts = [part.reshape(-1, 3) for part in (top, bottom, left, right) if part.size]
    if not parts:
        return (255, 255, 255)
    mean = np.concatenate(parts).mean(axis=0)
    lum = 0.299 * mean[2] + 0.587 * mean[1] + 0.114 * mean[0]
    return (20, 20, 20) if lum > 128 else (255, 255, 255)


def _text_line_boxes(
    img: np.ndarray,
    *,
    detect_threshold: int = 40,
    ratio_threshold: int = 50,
) -> list[dict[str, Any]]:
    """Detect horizontal text-line bounding boxes in the whole frame.

    Text strokes produce dense, small gradient blobs; long flat components
    (bar edges, axes, dividers) are dropped, and the remaining components are
    clustered into text lines by vertical overlap.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    mask = (grad > detect_threshold).astype(np.uint8) * 255
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = []
    for i in range(1, n):
        cx, cy, cw, ch, area = stats[i]
        if ch < 8 or ch > 120 or cw < 5 or area < 20:
            continue
        if cw > 10 * max(ch, 8):
            continue
        comps.append([int(cx), int(cy), int(cw), int(ch)])
    comps.sort(key=lambda c: c[1])
    lines: list[dict[str, int]] = []
    for cx, cy, cw, ch in comps:
        placed = False
        for ln in lines:
            if cy <= ln["y2"] + 2 and cy + ch >= ln["y1"] - 2:
                gap = max(0, cx - ln["x2"], ln["x1"] - (cx + cw))
                if gap <= 50:
                    ln["x1"] = min(ln["x1"], cx)
                    ln["x2"] = max(ln["x2"], cx + cw)
                    ln["y1"] = min(ln["y1"], cy)
                    ln["y2"] = max(ln["y2"], cy + ch)
                    placed = True
                    break
        if not placed:
            lines.append({"x1": cx, "y1": cy, "x2": cx + cw, "y2": cy + ch})
    for ln in lines:
        area = (ln["x2"] - ln["x1"]) * (ln["y2"] - ln["y1"])
        strong = (grad > ratio_threshold).astype(np.uint8) * 255
        ln["ratio"] = (
            float(strong[ln["y1"] : ln["y2"], ln["x1"] : ln["x2"]].sum() / 255) / area
            if area
            else 0.0
        )
    return lines


def _tighten_text_line(mask: np.ndarray, line: dict[str, int]) -> list[int]:
    """Narrow a text line to its dense text pixels (drops circle outlines,
    bar edges and other graphics that merged into the line)."""
    x1, y1, x2, y2 = line["x1"], line["y1"], line["x2"], line["y2"]
    sub = mask[y1:y2, x1:x2] > 0
    if sub.size == 0:
        return [x1, y1, x2, y2]
    col = sub.sum(axis=0)
    row = sub.sum(axis=1)
    cth = max(1, int(col.max() * 0.25))
    rth = max(1, int(row.max() * 0.25))
    cols = np.where(col >= cth)[0]
    rows = np.where(row >= rth)[0]
    if len(cols) == 0 or len(rows) == 0:
        return [x1, y1, x2, y2]
    return [
        x1 + int(cols.min()),
        y1 + int(rows.min()),
        x1 + int(cols.max()) + 1,
        y1 + int(rows.max()) + 1,
    ]


def locate_text_boxes(
    image_path: str | Path,
    aligned: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> dict[str, dict[str, list[int]]]:
    """Locate the original printed value/label text boxes for each (vertical)
    bar using pure CV geometry.

    Text lines are detected from gradient edges across the whole frame; for
    each bar the value box is the text line horizontally aligned with the bar
    whose bottom sits just above/at the bar top, and the label box is the
    line just below the baseline.  Deterministic, no extra model calls, and
    independent of text color or chart theme.
    """
    result: dict[str, dict[str, list[int]]] = {}
    img = cv2.imread(str(image_path))
    if img is None or not aligned:
        return result
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    textmask = (grad > 50).astype(np.uint8) * 255
    lines = _text_line_boxes(img)
    # Text components (connected strokes) used by both orientation branches:
    # matching components by bar overlap keeps one box per label, unlike the
    # merged text lines that can swallow adjacent category labels.
    n_comp, _, stats, _ = cv2.connectedComponentsWithStats(textmask, 8)
    comps = []
    for i in range(1, n_comp):
        cx, cy, cw, ch, area = [int(value) for value in stats[i]]
        if ch < 6 or ch > 130 or cw < 3 or area < 15:
            continue
        comps.append(
            {
                "x1": cx,
                "y1": cy,
                "x2": cx + cw,
                "y2": cy + ch,
                "ratio": float(area) / float(max(1, cw * ch)),
            }
        )

    def _span_hits(ln: dict[str, int], x: int, w: int) -> bool:
        center = (ln["x1"] + ln["x2"]) / 2
        return x - 15 <= center <= x + w + 15

    def _zone_lines(zone: tuple[int, int, int, int]) -> list[dict[str, int]]:
        zx1, zy1, zx2, zy2 = zone
        out = []
        for ln in lines:
            cx = (ln["x1"] + ln["x2"]) / 2
            cy = (ln["y1"] + ln["y2"]) / 2
            if zx1 - 25 <= cx <= zx2 + 25 and zy1 - 25 <= cy <= zy2 + 25:
                out.append(ln)
        return out

    def _center_dist(ln: dict[str, int], anchor: tuple[int, int]) -> float:
        ax, ay = anchor
        return ((ln["x1"] + ln["x2"]) / 2 - ax) ** 2 + ((ln["y1"] + ln["y2"]) / 2 - ay) ** 2

    for item in aligned:
        eid = str(item.get("entity_id") or "")
        x, y, w, hh = item.get("x"), item.get("y"), item.get("w"), item.get("h")
        if not eid or None in (x, y, w, hh):
            continue
        x, y, w, hh = int(x), int(y), int(w), int(hh)
        orientation = str(item.get("orientation") or ("horizontal" if w > hh else "vertical"))
        boxes = result.setdefault(eid, {})
        baseline = y + hh
        if orientation == "horizontal":
            # Horizontal bars: the value is printed at the right end of the
            # bar and the category label sits above the bar. Text lines keep
            # this branch stable (multi-line labels and long labels like
            # "Less than $20,000" are handled by line selection, not by
            # merging every overlapping component).
            value_candidates = _zone_lines((x + w - 140, y - 12, x + w + 180, y + hh + 12))
            label_candidates = _zone_lines((x - 12, max(0, y - 80), x + w + 12, y - 2))
            if value_candidates:
                solid = [ln for ln in value_candidates if ln["ratio"] >= 0.08]
                pool = solid or value_candidates
                ln = min(pool, key=lambda ln: _center_dist(ln, (x + w, y + hh // 2)))
                box = _tighten_text_line(textmask, ln)
                boxes["value_box"] = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
            if label_candidates:
                solid = [ln for ln in label_candidates if ln["ratio"] >= 0.08]
                pool = solid or label_candidates
                ln = max(pool, key=lambda ln: ln["y2"])
                box = _tighten_text_line(textmask, ln)
                boxes["label_box"] = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
        else:
            # Vertical bars: match text components by horizontal overlap with
            # the bar instead of merged text lines. Text lines merge adjacent
            # category labels (e.g. "Germany Switzerland United States" in one
            # box), while component-level matching keeps one box per label.
            # Value candidates must look like solid text (ratio >= 0.10) so
            # arrows, dashed grid lines and decorations never become value
            # boxes.
            def _overlap(a: dict[str, int], bx: int, bw: int) -> int:
                return max(0, min(a["x2"], bx + bw) - max(a["x1"], bx))

            value_zone_end = y + max(40, min(130, int(hh * 0.6)))
            value_comps = [
                comp
                for comp in comps
                if comp["y2"] >= y - 120
                and comp["y1"] <= value_zone_end
                and comp["ratio"] >= 0.10
                and _overlap(comp, x, w) >= max(4, int((comp["x2"] - comp["x1"]) * 0.35))
            ]
            if value_comps:
                box = _tighten_text_line(
                    textmask,
                    {
                        "x1": min(comp["x1"] for comp in value_comps),
                        "y1": min(comp["y1"] for comp in value_comps),
                        "x2": max(comp["x2"] for comp in value_comps),
                        "y2": max(comp["y2"] for comp in value_comps),
                        "ratio": 1.0,
                    },
                )
                boxes["value_box"] = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
            label_comps = [
                comp
                for comp in comps
                if comp["y1"] >= baseline + 2
                and comp["y1"] <= baseline + 110
                and _overlap(comp, x, w) >= max(4, int((comp["x2"] - comp["x1"]) * 0.35))
            ]
            if label_comps:
                box = _tighten_text_line(
                    textmask,
                    {
                        "x1": min(comp["x1"] for comp in label_comps),
                        "y1": min(comp["y1"] for comp in label_comps),
                        "x2": max(comp["x2"] for comp in label_comps),
                        "y2": max(comp["y2"] for comp in label_comps),
                        "ratio": 1.0,
                    },
                )
                boxes["label_box"] = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
    return result


def _render_overlay(
    image_path: str | Path,
    aligned: list[dict[str, Any]],
    out: Path,
    text_boxes: dict[str, dict[str, list[int]]] | None = None,
) -> bool:
    from PIL import Image, ImageDraw

    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    bar_color = (230, 25, 75)
    for a in aligned:
        d.rectangle([a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]], outline=bar_color, width=3)
        # Only the bar box is drawn.  Value/label boxes were removed: they
        # cluttered the review image and their positions are not reliable
        # (values are read by the vision model, not from these boxes).
    img.save(out)
    return out.exists()


def _render_aligned_svg(
    aligned: list[dict[str, Any]],
    out: Path,
    text_boxes: dict[str, dict[str, list[int]]] | None = None,
) -> bool:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" data-role="semantic-chart" data-generator="datavideo.cv_align_v1">',
    ]
    for i, a in enumerate(aligned):
        eid = re.sub(r"[^a-z0-9]+", "-", str(a.get("entity_id") or "").lower()).strip("-") or f"bar-{i}"
        value = a.get("value_text") or ""
        orientation = str(a.get("orientation") or ("horizontal" if a["w"] >= a["h"] else "vertical"))
        anim_prop = "width" if orientation == "horizontal" else "height"
        anchor = "left" if orientation == "horizontal" else "bottom"
        lines.append(f'<g id="entity-{eid}" data-role="entity" data-entity-id="{eid}" data-label="{html.escape(str(a.get("label") or ""))}">')
        lines.append(
            f'<rect id="{eid}-bar" data-role="bar" data-entity-id="{eid}" data-value="{html.escape(value)}" '
            f'x="{a["x"]}" y="{a["y"]}" width="{a["w"]}" height="{a["h"]}" '
            f'data-animation-property="{anim_prop}" data-anchor="{anchor}" data-orientation="{orientation}" fill="#3cb44b"/>'
        )
        if value:
            vw = max(52.0, _text_width_estimate(value, 20) + 18)
            if orientation == "horizontal":
                vx = a["x"] + a["w"] + 8
                vy = max(0, a["y"] + a["h"] / 2 - 14)
                text_anchor = "start"
                tx = vx + 9
            else:
                vx = a["x"] + a["w"] / 2 - vw / 2
                vy = max(0, a["y"] - 36)
                text_anchor = "middle"
                tx = a["x"] + a["w"] / 2
            # Value text only, no surrounding box (keeps the chart clean;
            # values come from the vision read, not from this geometry).
            lines.append(
                f'<text data-role="value-label" x="{tx:.1f}" y="{vy + 21}" '
                f'text-anchor="{text_anchor}" font-size="20" font-weight="700">{html.escape(value)}</text>'
            )
        label = str(a.get("label") or "")
        if label:
            # Category text is still emitted (it is the semantic label), but
            # the surrounding white box is dropped: it drew a rectangle around
            # every label and cluttered the chart.
            text_anchor = "middle"
            tx = a["x"] + a["w"] / 2
            if orientation == "horizontal":
                ly = max(0, a["y"] - 34)
                text_anchor = "start"
                tx = a["x"] + 4
            else:
                ly = a["y"] + a["h"] + 8
            lines.append(
                f'<text data-role="category-label" x="{tx:.1f}" y="{ly + 21}" '
                f'text-anchor="{text_anchor}" font-size="16">{html.escape(label)}</text>'
            )
        lines.append("</g>")
    lines.append("</svg>")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out.exists()


def run_cv_align(
    clip_id: str,
    image_path: str | Path,
    entities: list[dict[str, Any]],
    out_dir: str | Path,
    client: Any = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    boxes = detect_bars(image_path)
    orientation = boxes[0].get("orientation") if boxes else "vertical"
    # Standard data source: one structured table read per keyframe.  It
    # returns labels + printed values + in-frame title/unit in one shot, so
    # it replaces the separate entity-order read and the per-bar full-frame
    # value scans (fewer vision calls, consistent label/value pairing).
    table = read_chart_table(image_path, cfg, orientation)
    vision_order = [row["label"] for row in table["bars"]]
    if any(not label for label in vision_order):
        # The table read can miss a small / clipped label (e.g. the top bar
        # of bar_74).  Try a dedicated category-label read, then per-bar
        # label crops for anything still empty.
        try:
            order_fallback = read_entity_order(image_path, cfg, orientation)
        except Exception:
            order_fallback = []
        for idx, label in enumerate(vision_order):
            if label:
                continue
            if idx < len(order_fallback) and order_fallback[idx]:
                vision_order[idx] = order_fallback[idx]
            elif idx < len(boxes):
                try:
                    vision_order[idx] = read_bar_label(image_path, boxes[idx], cfg)
                except Exception:
                    pass
    aligned, warnings = match_entities(boxes, entities, vision_order)
    values = _assign_table_values(aligned, table)
    # Per-bar crop fallback for bars whose value the table could not verify
    # (single-read or absent); only when the table recovered at least one
    # value, so charts with no printed values go straight to tick estimation.
    any_table_value = any(item.get("value") is not None for item in values)
    if any_table_value:
        img = cv2.imread(str(image_path))
        for idx, item in enumerate(values):
            if item.get("value_read_verified") or img is None:
                continue
            crop = _crop_value_region(img, item)
            crop_path = Path(image_path).with_name(f"value_crop_{idx:02d}.png")
            cv2.imwrite(str(crop_path), crop)
            try:
                crop_text = _call_vision(
                    crop_path,
                    "读出这个裁剪图里的数字或百分比，只返回数字和单位，如 36.1% 或 6.9。",
                    cfg,
                    temperature=0.0,
                )
                match = _NUMBER_TOKEN_RE.search(crop_text)
                if match:
                    token = match.group(0).strip()
                    try:
                        num = float(token.replace(",", "").rstrip("%"))
                    except ValueError:
                        num = None
                    if num is not None:
                        values[idx] = {
                            **item,
                            "value": num,
                            "value_text": token,
                            "value_read_verified": True,
                        }
            except Exception:
                pass
    # A zero read on a visible bar is almost always a misread of an unlabeled
    # bar (or an axis tick), not a genuine zero; let the scale estimation
    # fill it in instead.
    for item in values:
        if item.get("value") == 0 and _bar_length(item) > 5 and str(item.get("value_text") or "") not in ("0%", "0 %"):
            item["value"] = None
            item["value_text"] = None
    # Values that survived the majority vote are treated as directly printed
    # (trusted); the scale estimation fills in whatever is still unlabeled.
    for item in values:
        if item.get("value_text"):
            item["value_read_verified"] = True
    # Drop frame-read values that are grossly inconsistent with the bar
    # geometry (e.g. a y-axis tick label like $30,000 misread as a bar's
    # value).  A real printed value tracks the bar's length ratio; a value
    # that is 10x off that ratio is a misread and should fall back to the
    # scale estimation instead of poisoning the calibration.
    verified = [float(v["value"]) for v in values if v.get("value_read_verified") and v.get("value")]
    lengths = [_bar_length(v) for v in values if _bar_length(v) > 0]
    if len(verified) >= 2 and lengths:
        max_verified = max(verified)
        max_length = max(lengths)
        for item in values:
            if not item.get("value_read_verified") or not item.get("value"):
                continue
            value_ratio = float(item["value"]) / max_verified
            length_ratio = _bar_length(item) / max_length
            if length_ratio > 0 and (value_ratio < 0.1 * length_ratio or value_ratio > 10.0 * length_ratio):
                item["value"] = None
                item["value_text"] = None
                item["value_read_verified"] = False
    estimated_count = estimate_unlabeled_values(values)
    tick_marks: list[dict[str, Any]] = []
    tick_unit = ""
    tick_estimated_count = 0
    verified_count = sum(1 for item in values if item.get("value_read_verified"))
    if verified_count < 2:
        # Not enough printed values for the labeled-bar scale: try calibrating
        # from the value-axis tick marks instead (e.g. unlabeled bar charts
        # that still draw a "$0/$100/$200" axis).
        try:
            tick_marks = detect_axis_tick_marks(image_path, orientation)
        except Exception:
            tick_marks = []
        if tick_marks:
            try:
                tick_labels, tick_unit = read_tick_labels(image_path, cfg, orientation)
            except Exception:
                tick_labels = []
            if tick_labels:
                paired = _pair_ticks_with_labels(tick_marks, tick_labels, orientation)
                if not paired and len(tick_labels) == len(tick_marks) + 1 and values:
                    # The 0/start anchor is usually hidden under the bars;
                    # recover it from the shared bar baseline.
                    baseline = _infer_baseline_coord(values, orientation)
                    if baseline is not None:
                        paired = _pair_ticks_with_labels(
                            [*tick_marks, {"coord": baseline}],
                            tick_labels,
                            orientation,
                        )
                tick_estimated_count = estimate_unlabeled_values_from_ticks(values, paired)
                # A directly-read value that strongly conflicts with the axis
                # scale (e.g. a $30,000 tick label misread as a bar value) is
                # a misread: clear it so the scale estimation takes over.
                scale = _tick_scale(tick_marks)
                if scale is not None:
                    for item in values:
                        if not item.get("value_read_verified") or not item.get("value"):
                            continue
                        if str(item.get("orientation") or "") == "horizontal":
                            coord = float(item.get("x") or 0.0) + _bar_length(item)
                        else:
                            coord = float(item.get("y") or 0.0)
                        expected = _estimate_coord_value(coord, scale)
                        if expected > 0 and abs(float(item["value"]) - expected) / expected > 2.0:
                            item["value"] = None
                            item["value_text"] = None
                            item["value_read_verified"] = False
    estimated_count += tick_estimated_count
    for item in values:
        item["value_plausible"], item["plausibility_message"] = _value_plausibility(item, values)
    implausible = [
        {"entity_id": item.get("entity_id"), "label": item.get("label"), "value_text": item.get("value_text"), "reason": item["plausibility_message"]}
        for item in values
        if not item["value_plausible"]
    ]
    consistent, message = _ratio_consistency(values)
    overlay = out_dir / "aligned_overlay.png"
    svg = out_dir / "semantic_aligned.svg"
    text_boxes: dict[str, dict[str, list[int]]] = {}
    try:
        text_boxes = locate_text_boxes(image_path, values, cfg)
    except Exception as exc:
        warnings.append(f"text box localization failed: {exc}")
    for item in values:
        item["value_box"] = (text_boxes.get(str(item.get("entity_id") or "")) or {}).get("value_box")
        item["label_box"] = (text_boxes.get(str(item.get("entity_id") or "")) or {}).get("label_box")
    overlay_ok = _render_overlay(image_path, values, overlay, text_boxes)
    svg_ok = _render_aligned_svg(values, svg, text_boxes)
    verified = False
    verify_message = ""
    if overlay_ok:
        try:
            verified, verify_message = verify_alignment(overlay, cfg)
        except Exception as exc:
            verify_message = f"vision verify failed: {exc}"
    report = {
        "clip_id": clip_id,
        "detected_bar_count": len(boxes),
        "matched_count": len(values),
        "estimated_value_count": estimated_count,
        "tick_estimated_value_count": tick_estimated_count,
        "tick_mark_count": len(tick_marks),
        "tick_unit": tick_unit,
        "orientation": (
            (values[0].get("orientation") if values else None)
            or (boxes[0].get("orientation") if boxes else None)
        ),
        "warnings": warnings,
        "value_geometry_consistent": consistent,
        "consistency_message": message,
        "implausible_bars": implausible,
        "value_read_method": "vision_standard_table",
        "standard_table": {
            "title": table.get("title") or "",
            "unit": table.get("unit") or "",
            "bars": table.get("bars") or [],
            "attempt_count": table.get("attempt_count") or 0,
        },
        "frame_title": table.get("title") or "",
        "frame_unit": table.get("unit") or "",
        "vision_entity_order": vision_order or [],
        "alignment_verified": verified,
        "alignment_verify_message": verify_message,
        "bars": values,
        "overlay_png": str(overlay),
        "aligned_svg": str(svg),
        "success": bool(values) and overlay_ok and svg_ok,
    }
    write_json(out_dir / "aligned_report.json", report)
    return report
