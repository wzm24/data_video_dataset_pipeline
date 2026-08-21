from pathlib import Path

import cv2
import numpy as np

from datavideo.cv_align import (
    _clean_vision_label,
    _contrast_outline_color,
    _labels_match,
    _labeled_value_pairs,
    _parse_tick_labels,
    _parse_label_json,
    _infer_baseline_coord,
    _assign_x_labels,
    bar_layout_regularity,
    _ratio_consistency,
    _render_aligned_svg,
    _render_overlay,
    _remove_horizontal_lines,
    _trace_dp_path,
    _value_plausibility,
    detect_axis_tick_marks,
    detect_bars,
    detect_lines,
    reconcile_line_dynamic,
    estimate_unlabeled_values,
    estimate_unlabeled_values_from_ticks,
    locate_text_boxes,
    match_entities,
)


BG = (112, 32, 240)  # BGR dark purple, like the WeChat test clip


def _synthetic_frame(tmp_path: Path) -> Path:
    img = np.full((720, 1280, 3), BG, dtype=np.uint8)
    bars = [
        (172, 236, (253, 208, 54), "36.1%", "Sub-Saharan Africa"),
        (432, 46, (78, 235, 129), "6.9%", "Latin America & Caribbean"),
        (692, 34, (52, 208, 193), "5.1%", "East Asia & Pacific"),
        (954, 6, (84, 168, 255), "1%", "European Union"),
    ]
    baseline = 566
    for x, height, color, value, label in bars:
        cv2.rectangle(img, (x, baseline - height), (x + 158, baseline), color, -1)
        cv2.putText(img, value, (x + 20, baseline - height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(img, label[:12], (x, baseline + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    path = tmp_path / "synthetic.png"
    cv2.imwrite(str(path), img)
    return path


def test_detect_bars_includes_short_bar(tmp_path):
    path = _synthetic_frame(tmp_path)
    boxes = detect_bars(path)
    assert len(boxes) == 4
    assert [b["x"] for b in boxes] == [172, 432, 692, 954]
    heights = [b["h"] for b in boxes]
    assert abs(heights[0] - 236) <= 3
    assert abs(heights[1] - 46) <= 3
    assert abs(heights[2] - 34) <= 3
    assert abs(heights[3] - 6) <= 3


def _tick_frame(tmp_path: Path, orientation: str = "vertical") -> Path:
    img = np.full((720, 1280, 3), 255, dtype=np.uint8)
    if orientation == "vertical":
        axis_x = 80
        cv2.line(img, (axis_x, 60), (axis_x, 660), (0, 0, 0), 3)
        for y, value in [(660, 0), (540, 100), (420, 200), (300, 300), (180, 400)]:
            cv2.line(img, (axis_x - 14, y), (axis_x + 14, y), (0, 0, 0), 3)
            cv2.putText(img, str(value), (20, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        # two bars whose tops sit between ticks: 200-300 range and 100-200 range
        cv2.rectangle(img, (200, 360), (360, 660), (90, 90, 250), -1)
        cv2.rectangle(img, (420, 480), (580, 660), (90, 200, 120), -1)
    else:
        axis_y = 640
        cv2.line(img, (80, axis_y), (1200, axis_y), (0, 0, 0), 3)
        for x, value in [(100, 0), (300, 100), (500, 200), (700, 300)]:
            cv2.line(img, (x, axis_y - 14), (x, axis_y + 14), (0, 0, 0), 3)
            cv2.putText(img, str(value), (x - 16, axis_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.rectangle(img, (120, 100), (550, 300), (90, 90, 250), -1)
        cv2.rectangle(img, (650, 380), (900, 560), (90, 200, 120), -1)
    path = tmp_path / f"ticks_{orientation}.png"
    cv2.imwrite(str(path), img)
    return path


def test_detect_axis_tick_marks_vertical(tmp_path):
    marks = detect_axis_tick_marks(_tick_frame(tmp_path, "vertical"), "vertical")
    coords = sorted(round(m["coord"]) for m in marks)
    assert len(coords) >= 4
    assert max(coords) > 600  # bottom tick near the axis end
    assert min(coords) < 250  # top tick


def test_detect_axis_tick_marks_horizontal(tmp_path):
    marks = detect_axis_tick_marks(_tick_frame(tmp_path, "horizontal"), "horizontal")
    coords = sorted(round(m["coord"]) for m in marks)
    assert len(coords) >= 3
    assert min(coords) < 200
    assert max(coords) > 500


def test_estimate_from_ticks_interpolates_values():
    aligned = [
        {"label": "A", "orientation": "vertical", "x": 200, "y": 360, "w": 160, "h": 300, "value": None},
        {"label": "B", "orientation": "vertical", "x": 420, "y": 480, "w": 160, "h": 180, "value": None},
    ]
    ticks = [
        {"coord": 660, "value": 0},
        {"coord": 540, "value": 100},
        {"coord": 420, "value": 200},
        {"coord": 300, "value": 300},
        {"coord": 180, "value": 400},
    ]
    count = estimate_unlabeled_values_from_ticks(aligned, ticks)
    assert count == 2
    assert abs(aligned[0]["value"] - 250.0) < 1e-6
    assert abs(aligned[1]["value"] - 150.0) < 1e-6
    assert aligned[0]["value_type"] == "estimated"


def test_parse_tick_labels():
    labels, unit = _parse_tick_labels('["$0", "$100", "$200"]')
    assert labels == [0.0, 100.0, 200.0]
    assert unit == "$"
    labels, unit = _parse_tick_labels('[0, 10, 20, 30]')
    assert labels == [0.0, 10.0, 20.0, 30.0]
    assert unit == ""
    labels, unit = _parse_tick_labels('["0%", "50%"]')
    assert unit == "%"
    assert _parse_tick_labels("no array here") == ([], "")
    # thousand/million suffixes are kept as the unit instead of silently
    # rescaling the value: "50k" stays 50 with unit "k"
    labels, unit = _parse_tick_labels('["50k", "100k", "350k"]')
    assert labels == [50.0, 100.0, 350.0]
    assert unit == "k"
    labels, unit = _parse_tick_labels('["$0", "$500k", "$1m"]')
    assert labels == [0.0, 500.0, 1.0]


def test_infer_baseline_coord_from_bar_geometry():
    bars = [
        {"orientation": "vertical", "x": 306, "y": 454, "w": 40, "h": 134},
        {"orientation": "vertical", "x": 920, "y": 194, "w": 40, "h": 394},
    ]
    assert _infer_baseline_coord(bars, "vertical") == 588.0
    horizontal = [
        {"orientation": "horizontal", "x": 120, "y": 100, "w": 400, "h": 40},
        {"orientation": "horizontal", "x": 220, "y": 200, "w": 700, "h": 40},
    ]
    assert _infer_baseline_coord(horizontal, "horizontal") == 120.0


def test_bar_layout_regularity_flags_cross_fade_frames():
    clean = [
        {"orientation": "vertical", "x": 386, "w": 78},
        {"orientation": "vertical", "x": 500, "w": 78},
        {"orientation": "vertical", "x": 614, "w": 78},
        {"orientation": "vertical", "x": 727, "w": 78},
        {"orientation": "vertical", "x": 841, "w": 78},
        {"orientation": "vertical", "x": 955, "w": 78},
    ]
    cross_fade = [
        {"orientation": "vertical", "x": 360, "w": 32},
        {"orientation": "vertical", "x": 464, "w": 48},
        {"orientation": "vertical", "x": 574, "w": 48},
        {"orientation": "vertical", "x": 625, "w": 52},
        {"orientation": "vertical", "x": 720, "w": 58},
        {"orientation": "vertical", "x": 812, "w": 74},
        {"orientation": "vertical", "x": 894, "w": 84},
    ]
    assert bar_layout_regularity(clean) > 0.9
    assert bar_layout_regularity(cross_fade) < 0.5
    assert bar_layout_regularity([]) == 1.0


def test_labels_match_distinguishes_insured_and_uninsured():
    assert _labels_match("Insured United States", "Insured United States") is True
    assert _labels_match("Uninsured United States", "Insured United States") is False
    assert _labels_match("Insured United States", "Uninsured United States") is False
    assert _labels_match("Less than $20,000", "Less than $20,000") is True
    assert _labels_match("United States", "Insured United States") is True
    assert _labels_match("UK", "United Kingdom") is False


def _line_frame(tmp_path: Path, *, thick: bool = False, curve: bool = False) -> Path:
    img = np.full((720, 1280, 3), 25, dtype=np.uint8)  # dark background
    width = 8 if thick else 3
    if curve:
        xs = np.linspace(220, 1060, 200)
        ys = 420 - 180 * np.sin((xs - 220) / 840 * np.pi * 1.5)
        for x, y in zip(xs, ys):
            cv2.circle(img, (int(x), int(y)), width, (255, 255, 255), -1)
        # fake x-axis ticks
        for tx in [300, 500, 700, 900]:
            cv2.line(img, (tx, 650), (tx, 670), (255, 255, 255), 2)
    else:
        points = [(220, 560), (430, 300), (640, 480), (850, 220), (1060, 380)]
        for index in range(len(points) - 1):
            cv2.line(img, points[index], points[index + 1], (255, 255, 255), width)
    path = tmp_path / "line.png"
    cv2.imwrite(str(path), img)
    return path


def test_detect_lines_finds_corners_on_thick_polyline(tmp_path):
    path = _line_frame(tmp_path, thick=True)
    lines = detect_lines(path)
    assert len(lines) == 1
    xs = [point[0] for point in lines[0]["points"]]
    assert min(xs) <= 240
    assert max(xs) >= 1040
    # thick line still yields the 5 corners
    assert 4 <= len(lines[0]["points"]) <= 6


def test_detect_lines_samples_smooth_curve_at_ticks(tmp_path):
    path = _line_frame(tmp_path, curve=True)
    lines = detect_lines(path)
    assert len(lines) == 1
    points = lines[0]["points"]
    assert len(points) >= 3
    tick_xs = [300, 500, 700, 900]
    detected_xs = [point[0] for point in points]
    # curve data points should include positions near the x-axis ticks
    near_ticks = sum(1 for tick in tick_xs if any(abs(dx - tick) < 40 for dx in detected_xs))
    assert near_ticks >= 2


def test_reconcile_line_dynamic_keeps_all_points():
    lines = [
        {
            "label": "Net additions",
            "points": [
                {"x": 182, "y": 442, "value": 133600.0, "x_label": "2000-01"},
                {"x": 539, "y": 324, "value": 221600.0, "x_label": "2007-08"},
            ],
        }
    ]
    dynamic = reconcile_line_dynamic(
        lines,
        clip_id="line_4",
        image_path="selected.png",
        keyframe_timestamp=5.0,
        unit="",
        series_label="Net additions",
    )
    assert dynamic["include_in_dataset"] is True
    assert len(dynamic["states"]) == 2
    assert len(dynamic["final_data_table"]) == 2
    assert all(row["value_type"] == "estimated" and row["needs_review"] for row in dynamic["states"])
    assert dynamic["states"][1]["value"] == 221600.0
    assert dynamic["states"][0]["state_key"] == "2000-01"
    assert dynamic["states"][1]["state_key"] == "2007-08"
    assert dynamic["states"][0]["entity"] == "Net additions"
    assert dynamic["states"][0]["entity_id"] == "net-additions"


def test_assign_x_labels_interpolates_years_and_snaps_categories():
    points = [{"x": 182}, {"x": 539}, {"x": 637}]
    _assign_x_labels(points, [182.0, 640.0, 1098.0], ["2000-01", "2010-11", "2018-09"], 182, 1097)
    assert points[0]["x_label"] == "2000-01"
    assert points[1]["x_label"] == "2007-08"
    assert points[2]["x_label"] == "2009-10"

    categorical = [{"x": 100}, {"x": 400}, {"x": 700}]
    _assign_x_labels(categorical, [], ["Jan", "Feb", "Mar"], 100, 700)
    assert categorical[0]["x_label"] == "Jan"
    assert categorical[1]["x_label"] == "Feb"
    assert categorical[2]["x_label"] == "Mar"


def test_remove_horizontal_lines_drops_gridline_rows():
    skeleton = np.zeros((80, 200), dtype=np.uint8)
    # polyline: diagonal with small wave
    for x in range(200):
        y = 20 + int(8 * np.sin(x / 18.0)) + x // 10
        skeleton[min(79, y), x] = 255
    # horizontal gridline remnant spanning almost every column
    skeleton[55, :] = 255
    cleaned = _remove_horizontal_lines(skeleton, min_share=0.6)
    assert cleaned[55].sum() == 0
    assert cleaned[20:45].sum() > 0


def test_trace_dp_path_follows_polyline_with_noise():
    skeleton = np.zeros((100, 240), dtype=np.uint8)
    for x in range(240):
        y = 30 + int(25 * np.sin(x / 30.0)) + 30
        skeleton[min(99, y), x] = 255
    # local noise blob (illustration-like) in the upper right, far from the
    # wave (wave y range is 35..85 there), so the DP must stay on the line
    skeleton[5:25, 180:215] = 255
    path = _trace_dp_path(skeleton)
    assert len(path) >= 200
    ys = [py for _, py in path]
    assert min(ys) >= 30
    assert max(ys) <= 90


def test_detect_bars_rejects_label_text(tmp_path):
    img = np.full((720, 1280, 3), BG, dtype=np.uint8)
    cv2.rectangle(img, (100, 300), (300, 566), (253, 208, 54), -1)
    cv2.putText(img, "Sub-Saharan Africa", (100, 610), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (253, 208, 54), 2)
    path = tmp_path / "label_only.png"
    cv2.imwrite(str(path), img)
    boxes = detect_bars(path)
    assert len(boxes) == 1
    assert abs(boxes[0]["h"] - 266) <= 2


def test_detect_bars_on_light_background(tmp_path):
    """Light-gray background with colored bars (news-graphic style) must not be
    mistaken for a saturated background that erases the bars."""
    img = np.full((720, 1280, 3), (216, 216, 216), dtype=np.uint8)
    salmon = (95, 112, 249)  # BGR
    cv2.rectangle(img, (272, 260), (606, 645), salmon, -1)
    cv2.rectangle(img, (666, 300), (1006, 645), salmon, -1)
    # value circles above the bars, plus category labels below
    cv2.circle(img, (439, 220), 34, (0, 215, 255), -1)
    cv2.circle(img, (836, 260), 34, (0, 215, 255), -1)
    cv2.putText(img, "cyclists", (320, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
    cv2.putText(img, "drivers", (740, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
    path = tmp_path / "light_bg.png"
    cv2.imwrite(str(path), img)

    boxes = detect_bars(path)
    assert len(boxes) == 2
    assert [b["x"] for b in boxes] == [272, 666]
    assert abs(boxes[0]["h"] - 385) <= 5
    assert abs(boxes[1]["h"] - 345) <= 5


def test_detect_bars_horizontal(tmp_path):
    """Horizontal bars share a left edge and vary in width; they must be
    detected with orientation='horizontal' and sorted top-to-bottom."""
    img = np.full((720, 1280, 3), (216, 216, 216), dtype=np.uint8)
    for cy, w, color in [(180, 380, (30, 144, 255)), (300, 240, (255, 165, 0)), (420, 120, (220, 20, 60))]:
        cv2.rectangle(img, (200, cy - 20), (200 + w, cy + 20), color, -1)
    path = tmp_path / "horizontal.png"
    cv2.imwrite(str(path), img)

    bars = detect_bars(path)
    assert len(bars) == 3
    assert all(b["orientation"] == "horizontal" for b in bars)
    assert [b["x"] for b in bars] == [200, 200, 200]
    assert all(abs(bars[i]["w"] - expected) <= 1 for i, expected in enumerate([380, 240, 120]))
    assert [b["y"] for b in bars] == sorted(b["y"] for b in bars)


def test_match_entities_uses_vision_order_and_creates_frame_entity():
    boxes = [
        {"x": 172, "y": 330, "w": 158, "h": 236},
        {"x": 432, "y": 520, "w": 158, "h": 46},
        {"x": 692, "y": 532, "w": 158, "h": 34},
        {"x": 954, "y": 560, "w": 158, "h": 6},
    ]
    entities = [
        {"id": "east-asia-pacific", "label": "East Asia & Pacific"},
        {"id": "european-union", "label": "European Union"},
        {"id": "latin-america-caribbean", "label": "Latin America & Caribbean"},
    ]
    vision_order = [
        "Sub-Saharan Africa",
        "Latin America & Caribbean",
        "East Asia & Pacific",
        "European Union",
    ]
    aligned, warnings = match_entities(boxes, entities, vision_order)
    assert [a["entity_id"] for a in aligned] == [
        "sub-saharan-africa",
        "latin-america-caribbean",
        "east-asia-pacific",
        "european-union",
    ]
    assert aligned[0]["entity_source"] == "frame"
    assert any("created entity from frame label" in w for w in warnings)


def test_match_entities_alias_matching():
    boxes = [{"x": 1, "y": 300, "w": 100, "h": 100}]
    entities = [{"id": "european-union", "label": "European Union"}]
    aligned, _ = match_entities(boxes, entities, vision_order=["EU"])
    assert aligned[0]["entity_id"] == "european-union"


def test_match_entities_falls_back_to_list_order():
    boxes = [
        {"x": 1, "y": 330, "w": 100, "h": 236},
        {"x": 2, "y": 520, "w": 100, "h": 46},
    ]
    entities = [
        {"id": "ssa", "label": "Sub-Saharan Africa"},
        {"id": "lac", "label": "Latin America & Caribbean"},
    ]
    aligned, _ = match_entities(boxes, entities)
    assert [a["entity_id"] for a in aligned] == ["ssa", "lac"]


def test_ratio_consistency_guards_and_detects_mismatch():
    assert _ratio_consistency([]) == (True, "too few bars to check")
    assert _ratio_consistency([{"label": "a", "h": 100, "value": 36.1}]) == (True, "too few bars to check")

    no_values = [
        {"label": "a", "h": 100, "value": None},
        {"label": "b", "h": 50, "value": None},
    ]
    ok, msg = _ratio_consistency(no_values)
    assert ok and "no numeric values" in msg

    degenerate = [
        {"label": "a", "h": 0, "value": 1.0},
        {"label": "b", "h": 0, "value": 2.0},
    ]
    ok, msg = _ratio_consistency(degenerate)
    assert ok and "degenerate" in msg

    mismatch = [
        {"label": "a", "h": 236, "value": 36.1},
        {"label": "b", "h": 46, "value": 30.0},
    ]
    ok, _ = _ratio_consistency(mismatch)
    assert not ok


def test_clean_vision_label():
    assert _clean_vision_label("1. Sub-Saharan Africa") == "Sub-Saharan Africa"
    assert _clean_vision_label("2) EU") == "EU"
    assert _clean_vision_label("  Latin America & Caribbean  ") == "Latin America & Caribbean"


def test_labeled_value_pairs_supports_dollar_and_comma_labels():
    text = "Less than $20,000: 890, More than $200,000: 1150"
    pairs = _labeled_value_pairs(text)
    assert ("Less than $20,000", "890") in pairs
    assert ("More than $200,000", "1150") in pairs


def test_parse_label_json_keeps_comma_labels_intact():
    text = (
        '```json\n'
        '[{"label": "Less than $20,000"}, {"label": "$40,000"},'
        ' {"label": "More than $200,000"}]\n'
        '```'
    )
    assert _parse_label_json(text) == [
        "Less than $20,000",
        "$40,000",
        "More than $200,000",
    ]


def test_render_aligned_svg_has_value_and_category_text(tmp_path):
    aligned = [
        {"x": 172, "y": 330, "w": 158, "h": 236, "entity_id": "ssa", "label": "Sub-Saharan Africa", "value_text": "36.1%"},
        {"x": 953, "y": 560, "w": 157, "h": 6, "entity_id": "eu", "label": "European Union", "value_text": "1%"},
    ]
    out = tmp_path / "aligned.svg"
    assert _render_aligned_svg(aligned, out)
    svg = out.read_text(encoding="utf-8")
    # Value and category text stay, but their surrounding boxes were removed
    # so the review chart shows only bars.
    assert 'data-role="value-box"' not in svg
    assert 'data-role="category-box"' not in svg
    assert 'data-role="value-label"' in svg
    assert 'data-role="category-label"' in svg
    assert "36.1%" in svg
    assert "European Union" in svg


def test_render_overlay_renders_with_boxes(tmp_path):
    img = np.full((720, 1280, 3), BG, dtype=np.uint8)
    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), img)
    aligned = [
        {
            "x": 172,
            "y": 330,
            "w": 158,
            "h": 236,
            "entity_id": "ssa",
            "label": "Sub-Saharan Africa",
            "value_text": "36.1%",
        }
    ]
    out = tmp_path / "overlay.png"
    assert _render_overlay(frame, aligned, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_overlay_only_bar_boxes(tmp_path):
    img = np.full((720, 1280, 3), BG, dtype=np.uint8)
    frame = tmp_path / "frame.png"
    cv2.imwrite(str(frame), img)
    aligned = [
        {
            "x": 172,
            "y": 330,
            "w": 158,
            "h": 236,
            "entity_id": "ssa",
            "label": "Sub-Saharan Africa",
            "value_text": "36.1%",
        }
    ]
    text_boxes = {
        "ssa": {"value_box": [180, 200, 320, 240], "label_box": [180, 600, 400, 640]}
    }
    out = tmp_path / "overlay.png"
    assert _render_overlay(frame, aligned, out, text_boxes)
    from PIL import Image

    rendered = Image.open(out).convert("RGB")
    # Only the bar box is drawn in red; value/label text boxes are no longer
    # rendered.
    assert rendered.getpixel((181, 220)) != (255, 255, 255)
    assert rendered.getpixel((181, 620)) != (255, 255, 255)
    assert rendered.getpixel((173, 400)) == (230, 25, 75)


def test_contrast_outline_color_adapts_to_background():
    dark = np.full((720, 1280, 3), BG, dtype=np.uint8)  # BGR purple
    light = np.full((720, 1280, 3), (216, 216, 216), dtype=np.uint8)  # BGR gray
    assert _contrast_outline_color(dark, [100, 100, 200, 140]) == (255, 255, 255)
    assert _contrast_outline_color(light, [100, 100, 200, 140]) == (20, 20, 20)


def test_locate_text_boxes_cv_fallback(tmp_path):
    img = np.full((720, 1280, 3), (216, 216, 216), dtype=np.uint8)
    cv2.rectangle(img, (272, 260), (606, 645), (95, 112, 249), -1)
    cv2.putText(img, "88%", (420, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(img, "cyclists", (330, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), img)
    aligned = [
        {"entity_id": "cyclists", "label": "cyclists", "x": 272, "y": 260, "w": 334, "h": 385}
    ]

    boxes = locate_text_boxes(path, aligned)
    value_box = boxes["cyclists"].get("value_box")
    label_box = boxes["cyclists"].get("label_box")
    assert value_box is not None and value_box[3] <= 270
    assert label_box is not None and label_box[1] >= 645


def test_locate_text_boxes_geometry_selection(tmp_path):
    """The value box must be the text line just above the bar and the label
    box the line just below the baseline, selected purely by geometry."""
    img = np.full((720, 1280, 3), (216, 216, 216), dtype=np.uint8)
    cv2.rectangle(img, (272, 260), (606, 645), (95, 112, 249), -1)
    cv2.putText(img, "88%", (400, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    cv2.putText(img, "cyclists", (330, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), img)
    aligned = [
        {
            "entity_id": "cyclists",
            "label": "cyclists",
            "x": 272,
            "y": 260,
            "w": 334,
            "h": 385,
            "value_text": "88%",
        }
    ]

    boxes = locate_text_boxes(path, aligned)
    value_box = boxes["cyclists"]["value_box"]
    label_box = boxes["cyclists"]["label_box"]
    assert value_box[3] < 260
    assert label_box is not None and label_box[1] >= 600


def test_locate_text_boxes_horizontal(tmp_path):
    """For horizontal bars the value is at the right end and the label sits
    above the bar."""
    img = np.full((720, 1280, 3), (216, 216, 216), dtype=np.uint8)
    cv2.rectangle(img, (200, 180), (580, 220), (30, 144, 255), -1)
    cv2.putText(img, "A", (210, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    cv2.putText(img, "380", (600, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), img)
    aligned = [
        {
            "entity_id": "a",
            "label": "A",
            "x": 200,
            "y": 180,
            "w": 380,
            "h": 40,
            "orientation": "horizontal",
            "value_text": "380",
        }
    ]
    boxes = locate_text_boxes(path, aligned)
    value_box = boxes["a"]["value_box"]
    label_box = boxes["a"]["label_box"]
    assert value_box is not None and value_box[0] >= 580
    assert label_box is not None and label_box[3] <= 180


def test_render_aligned_svg_horizontal(tmp_path):
    aligned = [
        {
            "x": 200,
            "y": 180,
            "w": 380,
            "h": 40,
            "entity_id": "a",
            "label": "A",
            "value_text": "380",
            "orientation": "horizontal",
        }
    ]
    out = tmp_path / "aligned.svg"
    assert _render_aligned_svg(aligned, out)
    svg = out.read_text(encoding="utf-8")
    assert 'data-animation-property="width"' in svg
    assert 'data-anchor="left"' in svg
    assert 'data-orientation="horizontal"' in svg


def test_value_plausibility_filters_outliers():
    aligned = [
        {"label": "SSA", "h": 236, "value": 36.1, "value_text": "36.1%"},
        {"label": "LAC", "h": 46, "value": 6.9, "value_text": "6.9%"},
        {"label": "EAP", "h": 34, "value": 5.1, "value_text": "5.1%"},
        {"label": "EU", "h": 6, "value": 1.0, "value_text": "1%"},
    ]
    for item in aligned:
        ok, _ = _value_plausibility(item, aligned)
        assert ok

    bad = {"label": "EU", "h": 6, "value": 248.0, "value_text": "248"}
    ok, message = _value_plausibility(bad, aligned)
    assert not ok
    assert "ratio" in message

    wrong = {"label": "SSA", "h": 236, "value": 30.0, "value_text": "30%"}
    ok, message = _value_plausibility(wrong, aligned)
    assert not ok
    assert "ratio" in message

    # Directly printed (majority-verified) values are trusted even when the
    # measured bar length disagrees slightly with the scale.
    printed = {"label": "SSA", "h": 456, "value": 890.0, "value_text": "890", "value_read_verified": True}
    ok, _ = _value_plausibility(printed, aligned)
    assert ok


def test_estimate_unlabeled_values_linear_scale_vertical():
    aligned = [
        {"label": "A", "h": 200, "w": 20, "value": 100.0, "value_text": "100"},
        {"label": "B", "h": 150, "w": 20, "value": None, "value_text": None},
        {"label": "C", "h": 50, "w": 20, "value": None, "value_text": None},
        {"label": "E", "h": 100, "w": 20, "value": 50.0, "value_text": "50"},
    ]
    count = estimate_unlabeled_values(aligned)
    assert count == 2
    by_label = {a["label"]: a for a in aligned}
    assert by_label["B"]["value_estimated"] is True
    assert by_label["B"]["value_type"] == "estimated"
    assert abs(by_label["B"]["value"] - 75.0) < 1.0
    assert abs(by_label["C"]["value"] - 25.0) < 1.0


def test_estimate_unlabeled_values_linear_scale_horizontal():
    aligned = [
        {"label": "A", "w": 400, "h": 30, "value": 1150.0, "value_text": "1150", "orientation": "horizontal"},
        {"label": "B", "w": 320, "h": 30, "value": None, "value_text": None, "orientation": "horizontal"},
        {"label": "D", "w": 300, "h": 30, "value": 890.0, "value_text": "890", "orientation": "horizontal"},
    ]
    count = estimate_unlabeled_values(aligned)
    assert count == 1
    by_label = {a["label"]: a for a in aligned}
    assert by_label["B"]["value_estimated"] is True
    # linear through (400,1150) and (300,890): slope = 2.6, intercept = 110
    assert abs(by_label["B"]["value"] - (110 + 2.6 * 320)) < 2.0
