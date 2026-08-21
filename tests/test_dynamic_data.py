from pathlib import Path

from datavideo.dynamic_data import (
    NO_RECOVERABLE_QUANTITATIVE_DATA,
    _label_embeds_own_value,
    sanitize_metric,
    visual_records_from_clip_data,
    build_dynamic_records,
    build_data_change_events,
    merge_consecutive_states,
    plan_dynamic_state_keyframes,
    plan_state_sampling,
)
from datavideo.multichart_qwen import _normalize_chart_data
from datavideo.multichart_assets import _select_clip_data_rows


def _visual_data(rows):
    return {
        "has_extractable_data": bool(rows),
        "chart_type": "bar",
        "unit": "%",
        "rows": rows,
        "manual_stub_rows": [],
        "visible_text": [str(row.get("evidence_text") or row.get("raw_text") or row.get("value")) for row in rows],
        "uncertain_fields": [],
    }


def _frame_context():
    return [
        {"image_index": 1, "source_frame": "frame_000", "time_seconds": 0.0},
        {"image_index": 2, "source_frame": "frame_001", "time_seconds": 0.5},
        {"image_index": 3, "source_frame": "frame_002", "time_seconds": 1.0},
    ]


def test_visual_only_values_are_included():
    result = build_dynamic_records(
        clip_id="bar_1",
        visual_data=_visual_data([{"label": "CAR", "value": "20%", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0}]),
        frame_context=_frame_context(),
        image_paths=["visual_frames/frame_000.jpg"],
        narration_sentences=[],
        chart_context={},
    )

    assert result["excluded"] is False
    assert result["states"][0]["source_type"] == "visual"
    assert result["states"][0]["entity_id"] == "car"
    assert result["states"][0]["value"] == 20.0


def test_sanitize_metric_drops_symbol_only_metrics():
    assert sanitize_metric("$$$") == ""
    assert sanitize_metric("!!!") == ""
    assert sanitize_metric("Illiteracy Rate") == "Illiteracy Rate"
    assert sanitize_metric(None) == ""


def test_label_embeds_own_value_detects_title_confusion():
    assert _label_embeds_own_value("380,000 km: Average Distance to the Moon", 380000.0) is True
    assert _label_embeds_own_value("Men: 50%", 50.0) is False
    assert _label_embeds_own_value("Income: $40,000", 40000.0) is False
    assert _label_embeds_own_value("Sub-Saharan Africa", 48.0) is False


def test_visual_rows_with_embedded_label_values_are_dropped():
    result = build_dynamic_records(
        clip_id="bar_1",
        visual_data=_visual_data(
            [
                {"label": "380,000 km: Average Distance to the Moon", "value": "380000 km", "unit": "km", "source_frame": "frame_000", "time_seconds": 0.0},
                {"label": "CAR", "value": "1 km", "unit": "km", "source_frame": "frame_000", "time_seconds": 0.0},
            ]
        ),
        frame_context=_frame_context(),
        image_paths=["visual_frames/frame_000.jpg"],
        narration_sentences=[],
        chart_context={},
    )
    entities = {row["entity"] for row in result["states"]}
    assert "380,000 km: Average Distance to the Moon" not in entities
    assert "CAR" in entities


def test_visual_records_drop_axis_ticks_shared_by_all_entities():
    """0/50/100 recovered for every entity at the same timestamp are axis
    ticks, not data, and must not become dynamic records."""
    data = _visual_data(
        [
            {"label": "cyclists", "value": "0", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
            {"label": "cyclists", "value": "50", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
            {"label": "cyclists", "value": "100", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
            {"label": "drivers", "value": "0", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
            {"label": "drivers", "value": "50", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
            {"label": "drivers", "value": "100", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
            {"label": "cyclists", "value": "88%", "unit": "%", "source_frame": "frame_002", "time_seconds": 1.0},
            {"label": "drivers", "value": "85%", "unit": "%", "source_frame": "frame_002", "time_seconds": 1.0},
        ]
    )
    records = visual_records_from_clip_data(
        data,
        _frame_context(),
        ["visual_frames/frame_000.jpg", "visual_frames/frame_001.jpg", "visual_frames/frame_002.jpg"],
        "combined_2",
    )
    by_key = {(r["entity_id"], r["state_start"]): r["value"] for r in records}
    assert set(by_key) == {("cyclists", 1.0), ("drivers", 1.0)}
    assert by_key[("cyclists", 1.0)] == 88.0
    assert by_key[("drivers", 1.0)] == 85.0


def test_narration_only_values_are_included():
    result = build_dynamic_records(
        clip_id="bar_1",
        visual_data=_visual_data([]),
        frame_context=[],
        image_paths=[],
        narration_sentences=[{"sentence_index": 7, "text": "CAR reaches 80 miles.", "confidence": 0.91, "start": 0.0, "end": 2.0}],
        chart_context={"chart_metadata": {"series": [{"label": "CAR"}]}},
    )

    assert result["excluded"] is False
    assert result["states"][0]["source_type"] == "narration"
    assert result["states"][0]["evidence_sentence_id"] == "7"
    assert result["states"][0]["value"] == 80.0
    assert result["states"][0]["unit"] == "miles"


def test_no_visual_or_narration_values_excludes_clip():
    result = build_dynamic_records(
        clip_id="bar_1",
        visual_data=_visual_data([]),
        frame_context=[],
        image_paths=[],
        narration_sentences=[{"text": "The car increases quickly.", "start": 0.0, "end": 1.0}],
        chart_context={"chart_metadata": {"series": [{"label": "CAR"}]}},
    )

    assert result["excluded"] is True
    assert result["exclude_reason"] == NO_RECOVERABLE_QUANTITATIVE_DATA
    assert result["states"] == []


def test_visual_narration_conflict_is_marked_needs_review():
    result = build_dynamic_records(
        clip_id="bar_1",
        visual_data=_visual_data([{"label": "CAR", "value": "20", "source_frame": "frame_000", "time_seconds": 0.0}]),
        frame_context=_frame_context(),
        image_paths=["visual_frames/frame_000.jpg"],
        narration_sentences=[{"sentence_index": 2, "text": "CAR is 30%.", "confidence": 0.9, "start": 0.0, "end": 1.0}],
        chart_context={"chart_metadata": {"series": [{"label": "CAR"}]}},
    )

    assert len(result["states"]) == 2
    assert {row["review_status"] for row in result["states"]} == {"needs_review"}
    assert {row["source_type"] for row in result["states"]} == {"visual", "narration"}


def test_consecutive_repeated_states_merge():
    states = merge_consecutive_states(
        [
            {"clip_id": "c", "entity_id": "car", "metric": "value", "value": 10, "unit": "%", "state_start": 0.0, "state_end": 0.5, "source_type": "visual", "evidence_frames": [], "confidence": 0.8, "review_status": "machine"},
            {"clip_id": "c", "entity_id": "car", "metric": "value", "value": 10, "unit": "%", "state_start": 0.5, "state_end": 1.0, "source_type": "visual", "evidence_frames": [], "confidence": 0.8, "review_status": "machine"},
        ]
    )

    assert len(states) == 1
    assert states[0]["state_start"] == 0.0
    assert states[0]["state_end"] == 1.0


def test_a_b_a_keeps_three_states():
    states = merge_consecutive_states(
        [
            {"clip_id": "c", "entity_id": "car", "metric": "value", "value": 10, "unit": "%", "state_start": 0.0, "state_end": 0.5, "source_type": "visual", "evidence_frames": [], "confidence": 0.8, "review_status": "machine"},
            {"clip_id": "c", "entity_id": "car", "metric": "value", "value": 20, "unit": "%", "state_start": 0.5, "state_end": 1.0, "source_type": "visual", "evidence_frames": [], "confidence": 0.8, "review_status": "machine"},
            {"clip_id": "c", "entity_id": "car", "metric": "value", "value": 10, "unit": "%", "state_start": 1.0, "state_end": 1.5, "source_type": "visual", "evidence_frames": [], "confidence": 0.8, "review_status": "machine"},
        ]
    )

    assert [row["value"] for row in states] == [10, 20, 10]
    assert [row["state_id"] for row in states] == ["state_001", "state_002", "state_003"]


def test_pure_animation_interpolation_with_unchanged_data_saves_one_state():
    result = build_dynamic_records(
        clip_id="bar_1",
        visual_data=_visual_data(
            [
                {"label": "CAR", "value": "20", "source_frame": "frame_000", "time_seconds": 0.0},
                {"label": "CAR", "value": "20", "source_frame": "frame_001", "time_seconds": 0.5},
                {"label": "CAR", "value": "20", "source_frame": "frame_002", "time_seconds": 1.0},
            ]
        ),
        frame_context=_frame_context(),
        image_paths=["visual_frames/frame_000.jpg", "visual_frames/frame_001.jpg", "visual_frames/frame_002.jpg"],
        narration_sentences=[],
        chart_context={},
    )

    assert len(result["states"]) == 1
    assert result["states"][0]["state_start"] == 0.0
    assert result["states"][0]["state_end"] == 1.0


def test_two_fps_coarse_scan_triggers_local_eight_fps_sampling():
    rows = [
        {"frame_id": "f0", "timestamp": 0.0, "score": {"motion_score": 0.0}},
        {"frame_id": "f1", "timestamp": 0.125, "score": {"motion_score": 0.0}},
        {"frame_id": "f2", "timestamp": 0.25, "score": {"motion_score": 0.0}},
        {"frame_id": "f3", "timestamp": 0.5, "score": {"motion_score": 0.4}},
    ]

    plan = plan_state_sampling(rows, {"clip_data": {"coarse_fps": 2, "fine_fps": 8, "motion_change_threshold": 0.1}})

    assert plan["coarse_fps"] == 2.0
    assert plan["fine_windows"] == [{"start": 0.0, "end": 0.5, "target_fps": 8.0, "reason": "coarse_motion_change"}]
    assert [row["frame_id"] for row in plan["selected_rows"]] == ["f0", "f1", "f2", "f3"]


def test_state_sampling_keeps_tail_frame_for_final_state():
    rows = [
        {"frame_id": "f0", "timestamp": 0.0, "score": {"motion_score": 0.0}},
        {"frame_id": "f1", "timestamp": 0.5, "score": {"motion_score": 0.0}},
        {"frame_id": "f2", "timestamp": 1.0, "score": {"motion_score": 0.0}},
        {"frame_id": "f3", "timestamp": 1.5, "score": {"motion_score": 0.0}},
        {"frame_id": "f_tail", "timestamp": 1.75, "score": {"motion_score": 0.0}},
    ]

    plan = plan_state_sampling(rows, {"clip_data": {"coarse_fps": 2, "fine_fps": 8, "motion_change_threshold": 0.1}})

    assert [row["frame_id"] for row in plan["selected_rows"]] == ["f0", "f1", "f2", "f3", "f_tail"]


def test_clip_data_row_selection_keeps_tail_when_limited():
    rows = [
        {
            "frame_id": f"f{idx}",
            "timestamp": idx * 0.25,
            "score": {
                "target_chart_type_match": True,
                "scene_change_or_title_card": False,
                "structure_complete": True,
                "edge_crop_or_occlusion": False,
            },
        }
        for idx in range(8)
    ]

    selected = _select_clip_data_rows(rows, [], {"clip_data": {"max_frames": 4}})

    assert selected[0]["frame_id"] == "f0"
    assert selected[-1]["frame_id"] == "f7"


def test_qwen_and_asr_failure_keeps_audit_without_forging_data():
    result = build_dynamic_records(
        clip_id="bar_1",
        visual_data=_visual_data([]),
        frame_context=[],
        image_paths=[],
        narration_sentences=[],
        chart_context={"chart_metadata": {"series": [{"label": "CAR"}]}},
        audit=[
            {"stage": "visual_qwen", "model_status": "unavailable", "failure_reason": "gpu unavailable"},
            {"stage": "narration", "status": "missing", "reason": "asr missing"},
        ],
    )

    assert result["excluded"] is True
    assert result["exclude_reason"] == NO_RECOVERABLE_QUANTITATIVE_DATA
    assert result["states"] == []
    assert result["audit"][0]["failure_reason"] == "gpu unavailable"
    assert result["audit"][1]["status"] == "missing"


def test_bar_one_narration_partial_dynamic_table_is_included():
    result = build_dynamic_records(
        clip_id="bar_1",
        visual_data={
            "rows": [],
            "visible_text": ["BOING 747", "SPACESHIP"],
            "series": [{"label": "Boeing 747"}, {"label": "Spaceship"}],
            "x_axis": "Boeing 747",
            "y_axis": "Spaceship",
        },
        frame_context=[
            {"image_index": 1, "source_frame": "frame_002", "time_seconds": 2.0},
            {"image_index": 2, "source_frame": "frame_009", "time_seconds": 9.0},
        ],
        image_paths=["frame_002.jpg", "frame_009.jpg"],
        narration_sentences=[
            {
                "sentence_id": "sent_001",
                "text": "A 747 would need 28 straight days to fly to the moon, and even with our current technology, we'd need two full days.",
                "start_context": 0.74,
                "end_context": 8.64,
                "confidence": 0.95,
            },
            {
                "sentence_id": "sent_002",
                "text": "And in a car, a lot of time.",
                "start_context": 9.0,
                "end_context": 11.66,
                "confidence": 0.95,
            },
        ],
        chart_context={"chart_metadata": {"series": [{"label": "Boeing 747"}, {"label": "Spaceship"}]}},
    )

    assert result["include_in_dataset"] is True
    assert result["data_completeness"] == "partial"
    assert result["numeric_fact_count"] == 2
    assert result["dynamic_data"] is True
    assert result["data_change_count"] == 3
    table = {row["entity"]: row for row in result["final_data_table"]}
    assert table["BOING 747"]["value"] == 28.0
    assert table["BOING 747"]["unit"] == "day"
    assert table["SPACESHIP"]["value"] == 2.0
    assert table["SPACESHIP"]["unit"] == "day"
    assert table["CAR"]["value"] is None
    assert table["CAR"]["type"] == 'qualitative: "a lot of time"'
    assert [event["event_type"] for event in result["data_change_events"]] == ["insert", "insert", "insert"]


def test_data_change_events_update_when_value_changes_and_skip_repeats():
    states = merge_consecutive_states(
        [
            {"clip_id": "c", "entity_id": "car", "entity": "CAR", "metric": "value", "value": 10, "unit": "%", "value_type": "exact", "state_start": 0.0, "state_end": 0.5, "source_type": "visual", "evidence_frames": [], "confidence": 0.8, "review_status": "machine"},
            {"clip_id": "c", "entity_id": "car", "entity": "CAR", "metric": "value", "value": 10, "unit": "%", "value_type": "exact", "state_start": 0.5, "state_end": 1.0, "source_type": "visual", "evidence_frames": [], "confidence": 0.8, "review_status": "machine"},
            {"clip_id": "c", "entity_id": "car", "entity": "CAR", "metric": "value", "value": 20, "unit": "%", "value_type": "exact", "state_start": 1.0, "state_end": 1.5, "source_type": "visual", "evidence_frames": [], "confidence": 0.8, "review_status": "machine"},
        ]
    )

    events = build_data_change_events(states)

    assert [event["event_type"] for event in events] == ["insert", "update"]


def test_rows_with_same_visible_state_share_state_id():
    result = build_dynamic_records(
        clip_id="bar_2",
        visual_data=_visual_data(
            [
                {"state": "1990", "year": 1990, "label": "Sub-Saharan Africa", "value": "48 %", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
                {"state": "1990", "year": 1990, "label": "Latin America & Caribbean", "value": "15.5 %", "unit": "%", "source_frame": "frame_001", "time_seconds": 0.5},
                {"state": "1990", "year": 1990, "label": "East Asia & Pacific", "value": "18 %", "unit": "%", "source_frame": "frame_002", "time_seconds": 1.0},
            ]
        ),
        frame_context=_frame_context(),
        image_paths=["visual_frames/frame_000.jpg", "visual_frames/frame_001.jpg", "visual_frames/frame_002.jpg"],
        narration_sentences=[],
        chart_context={},
    )

    assert {row["state_id"] for row in result["states"]} == {"state_001"}
    assert {row["state_key"] for row in result["states"]} == {"1990"}
    assert {row["state_label"] for row in result["states"]} == {"1990"}
    assert {row["state_start"] for row in result["states"]} == {0.0}
    assert {row["state_end"] for row in result["states"]} == {1.0}


def test_visual_rows_use_chart_y_axis_as_metric_before_entity_series():
    result = build_dynamic_records(
        clip_id="bar_2",
        visual_data={
            **_visual_data(
                [
                    {"state": "1990", "year": 1990, "label": "Sub-Saharan Africa", "series": "Sub-Saharan Africa", "y": 0, "value": "48 %", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
                ]
            ),
            "y_axis": "Illiteracy Rate",
        },
        frame_context=_frame_context(),
        image_paths=["visual_frames/frame_000.jpg"],
        narration_sentences=[],
        chart_context={},
    )

    assert result["states"][0]["metric"] == "Illiteracy Rate"


def test_dynamic_state_keyframes_select_first_and_last_complete_states():
    result = build_dynamic_records(
        clip_id="bar_2",
        visual_data=_visual_data(
            [
                {"state": "1990", "year": 1990, "label": "Sub-Saharan Africa", "value": "48 %", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
                {"state": "1990", "year": 1990, "label": "Latin America & Caribbean", "value": "15.5 %", "unit": "%", "source_frame": "frame_001", "time_seconds": 0.5},
                {"state": "2017", "year": 2017, "label": "Sub-Saharan Africa", "value": "35.9 %", "unit": "%", "source_frame": "frame_002", "time_seconds": 1.0},
                {"state": "2017", "year": 2017, "label": "Latin America & Caribbean", "value": "6.8 %", "unit": "%", "source_frame": "frame_003", "time_seconds": 1.5},
            ]
        ),
        frame_context=_frame_context(),
        image_paths=["visual_frames/frame_000.jpg", "visual_frames/frame_001.jpg", "visual_frames/frame_002.jpg", "visual_frames/frame_003.jpg"],
        narration_sentences=[],
        chart_context={},
    )

    plan = plan_dynamic_state_keyframes(result)

    assert plan["should_save"] is True
    assert [state["state_id"] for state in plan["states"]] == ["state_001", "state_002"]


def test_dynamic_state_keyframes_skip_single_state():
    result = build_dynamic_records(
        clip_id="bar_2",
        visual_data=_visual_data(
            [
                {"state": "1990", "year": 1990, "label": "Sub-Saharan Africa", "value": "48 %", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
            ]
        ),
        frame_context=_frame_context(),
        image_paths=["visual_frames/frame_000.jpg"],
        narration_sentences=[],
        chart_context={},
    )

    plan = plan_dynamic_state_keyframes(result)

    assert plan["should_save"] is False


def test_dynamic_state_keyframes_keep_all_complete_states():
    result = build_dynamic_records(
        clip_id="bar_3",
        visual_data=_visual_data(
            [
                {"state": "1990", "year": 1990, "label": "A", "value": "10 %", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
                {"state": "2000", "year": 2000, "label": "A", "value": "20 %", "unit": "%", "source_frame": "frame_001", "time_seconds": 0.5},
                {"state": "2010", "year": 2010, "label": "A", "value": "15 %", "unit": "%", "source_frame": "frame_002", "time_seconds": 1.0},
            ]
        ),
        frame_context=_frame_context(),
        image_paths=["visual_frames/frame_000.jpg", "visual_frames/frame_001.jpg", "visual_frames/frame_002.jpg"],
        narration_sentences=[],
        chart_context={},
    )

    plan = plan_dynamic_state_keyframes(result)

    assert plan["should_save"] is True
    assert [state["state_id"] for state in plan["states"]] == ["state_001", "state_002", "state_003"]


def test_static_line_points_from_one_frame_do_not_create_state_keyframes():
    result = build_dynamic_records(
        clip_id="line_32",
        visual_data={
            "has_extractable_data": True,
            "chart_type": "line",
            "unit": "%",
            "x_axis": "Year",
            "y_axis": "Share of income",
            "rows": [
                {
                    "year": year,
                    "label": "Top 10%",
                    "x": str(year),
                    "value": f"{value} %",
                    "unit": "%",
                    "source_frame": "frame_tail",
                    "time_seconds": 3.95,
                }
                for year, value in [(1910, 40), (1920, 42), (1930, 45), (2010, 48)]
            ],
            "manual_stub_rows": [],
            "visible_text": ["1910", "1920", "1930", "2010", "40%", "42%", "45%", "48%"],
            "uncertain_fields": [],
        },
        frame_context=[{"image_index": 1, "source_frame": "frame_tail", "time_seconds": 3.95}],
        image_paths=["visual_frames/frame_tail.jpg"],
        narration_sentences=[],
        chart_context={},
    )

    assert result["dynamic_data"] is True
    plan = plan_dynamic_state_keyframes(result)

    assert plan["should_save"] is False
    assert plan["reason"] == "static_chart_points_from_single_visual_frame"


def test_static_grouped_bar_marks_from_one_frame_do_not_create_state_keyframes():
    result = build_dynamic_records(
        clip_id="bar_6",
        visual_data=_visual_data(
            [
                {"label": "Auto", "metric": "National average", "value": "75%", "unit": "%", "source_frame": "frame_tail", "time_seconds": 8.95},
                {"label": "Auto", "metric": "Transit-oriented developments", "value": "45%", "unit": "%", "source_frame": "frame_tail", "time_seconds": 8.95},
                {"label": "Bike", "metric": "National average", "value": "2%", "unit": "%", "source_frame": "frame_tail", "time_seconds": 8.95},
                {"label": "Bike", "metric": "Transit-oriented developments", "value": "5%", "unit": "%", "source_frame": "frame_tail", "time_seconds": 8.95},
            ]
        ),
        frame_context=[{"image_index": 1, "source_frame": "frame_tail", "time_seconds": 8.95}],
        image_paths=["visual_frames/frame_tail.jpg"],
        narration_sentences=[],
        chart_context={},
    )

    assert result["dynamic_data"] is True
    assert {row["state_key"] for row in result["states"]} == {None}
    plan = plan_dynamic_state_keyframes(result)

    assert plan["should_save"] is False
    assert plan["reason"] == "no_explicit_state_labels"


def test_static_grouped_bar_metrics_are_kept_in_final_table_and_events():
    result = build_dynamic_records(
        clip_id="bar_6",
        visual_data=_visual_data(
            [
                {"label": "Auto", "metric": "National average", "value": "75%", "unit": "%", "source_frame": "frame_tail", "time_seconds": 8.95},
                {"label": "Auto", "metric": "Transit-oriented developments", "value": "45%", "unit": "%", "source_frame": "frame_tail", "time_seconds": 8.95},
                {"label": "Bike", "metric": "National average", "value": "2%", "unit": "%", "source_frame": "frame_tail", "time_seconds": 8.95},
                {"label": "Bike", "metric": "Transit-oriented developments", "value": "5%", "unit": "%", "source_frame": "frame_tail", "time_seconds": 8.95},
            ]
        ),
        frame_context=[{"image_index": 1, "source_frame": "frame_tail", "time_seconds": 8.95}],
        image_paths=["visual_frames/frame_tail.jpg"],
        narration_sentences=[],
        chart_context={},
    )

    table_keys = {(row["entity_id"], row["metric"]) for row in result["final_data_table"]}
    event_keys = {(row["entity_id"], row["metric"]) for row in result["data_change_events"]}

    assert len(result["final_data_table"]) == 4
    assert table_keys == {
        ("auto", "National average"),
        ("auto", "Transit-oriented developments"),
        ("bike", "National average"),
        ("bike", "Transit-oriented developments"),
    }
    assert event_keys == table_keys


def test_dynamic_state_keyframes_cap_keeps_first_and_last():
    rows = []
    frames = []
    for index in range(12):
        rows.append(
            {
                "state": str(1990 + index),
                "year": 1990 + index,
                "label": "A",
                "value": f"{10 + index} %",
                "unit": "%",
                "source_frame": f"frame_{index:03d}",
                "time_seconds": index * 0.5,
            }
        )
        frames.append({"image_index": index + 1, "source_frame": f"frame_{index:03d}", "time_seconds": index * 0.5})
    result = build_dynamic_records(
        clip_id="bar_4",
        visual_data=_visual_data(rows),
        frame_context=frames,
        image_paths=[f"visual_frames/frame_{index:03d}.jpg" for index in range(12)],
        narration_sentences=[],
        chart_context={},
    )

    plan = plan_dynamic_state_keyframes(result, max_states=4)

    assert plan["should_save"] is True
    keys = [state["state_id"] for state in plan["states"]]
    assert len(keys) == 4
    assert keys[0] == "state_001"
    assert keys[-1] == "state_012"


def test_bar_two_start_and_end_years_are_kept_as_dynamic_updates():
    rows = [
        {"state": "1990", "year": 1990, "label": "Sub-Saharan Africa", "value": "48 %", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
        {"state": "1990", "year": 1990, "label": "Latin America & Caribbean", "value": "15.5 %", "unit": "%", "source_frame": "frame_000", "time_seconds": 0.0},
        {"state": "2017", "year": 2017, "label": "Sub-Saharan Africa", "value": "35.9 %", "unit": "%", "source_frame": "frame_002", "time_seconds": 1.0},
        {"state": "2017", "year": 2017, "label": "Latin America & Caribbean", "value": "6.8 %", "unit": "%", "source_frame": "frame_002", "time_seconds": 1.0},
    ]
    result = build_dynamic_records(
        clip_id="bar_2",
        visual_data=_visual_data(rows),
        frame_context=_frame_context(),
        image_paths=["visual_frames/frame_000.jpg", "visual_frames/frame_001.jpg", "visual_frames/frame_002.jpg"],
        narration_sentences=[],
        chart_context={},
    )

    assert [row["state_id"] for row in result["states"]] == ["state_001", "state_001", "state_002", "state_002"]
    table = {row["entity_id"]: row for row in result["final_data_table"]}
    assert table["sub-saharan-africa"]["value"] == 35.9
    assert table["latin-america-caribbean"]["value"] == 6.8
    assert [event["event_type"] for event in result["data_change_events"]] == ["insert", "insert", "update", "update"]


def test_grouped_qwen_states_are_flattened_without_losing_year():
    data = _normalize_chart_data(
        {
            "has_extractable_data": True,
            "chart_type": "bar",
            "states": [
                {
                    "state": "1990",
                    "year": 1990,
                    "source_frame": "frame_000",
                    "time_seconds": 0.0,
                    "rows": [
                        {"label": "Sub-Saharan Africa", "value": "48 %", "unit": "%", "raw_text": "48 %"},
                    ],
                },
                {
                    "state": "2017",
                    "year": 2017,
                    "source_frame": "frame_002",
                    "time_seconds": 1.0,
                    "rows": [
                        {"label": "Sub-Saharan Africa", "value": "35.9 %", "unit": "%", "raw_text": "35.9 %"},
                    ],
                },
            ],
            "visible_text": ["48 %", "35.9 %"],
        },
        "bar",
    )

    assert data["has_extractable_data"] is True
    assert [(row["state"], row["year"], row["value"], row["source_frame"]) for row in data["rows"]] == [
        ("1990", 1990, "48 %", "frame_000"),
        ("2017", 2017, "35.9 %", "frame_002"),
    ]


def test_qwen_rows_are_deduped_when_series_and_states_repeat_same_value():
    data = _normalize_chart_data(
        {
            "has_extractable_data": True,
            "chart_type": "bar",
            "states": [
                {
                    "state": "Sub-Saharan Africa",
                    "year": 1990,
                    "source_frame": "frame_000",
                    "time_seconds": 0.0,
                    "rows": [
                        {"state": "Sub-Saharan Africa", "year": 1990, "label": "Sub-Saharan Africa", "value": "48 %", "unit": "%", "raw_text": "48 %"},
                    ],
                },
            ],
            "rows": [
                {"state": "Sub-Saharan Africa", "year": 1990, "label": "Sub-Saharan Africa", "value": "48 %", "unit": "%", "raw_text": "48 %", "source_frame": "frame_000", "time_seconds": 0.0},
            ],
            "visible_text": ["48 %"],
        },
        "bar",
    )

    assert len(data["rows"]) == 1
    result = build_dynamic_records(
        clip_id="bar_2",
        visual_data=data,
        frame_context=[{"image_index": 1, "source_frame": "frame_000", "time_seconds": 0.0}],
        image_paths=["visual_frames/frame_000.jpg"],
        narration_sentences=[],
        chart_context={},
    )
    assert result["states"][0]["state_key"] == "1990"


def test_qwen_title_entity_rows_are_filtered():
    data = _normalize_chart_data(
        {
            "has_extractable_data": True,
            "chart_type": "bar",
            "title": "Illiteracy Rate 1990",
            "y_axis": "Illiteracy Rate",
            "rows": [
                {"state": "Illiteracy Rate 1990", "year": 1990, "label": "Illiteracy Rate 1990", "value": "48 %", "unit": "%", "raw_text": "48 %"},
                {"state": "Sub-Saharan Africa", "year": 1990, "label": "Sub-Saharan Africa", "value": "48 %", "unit": "%", "raw_text": "48 %"},
            ],
            "visible_text": ["Illiteracy Rate 1990", "48 %"],
        },
        "bar",
    )

    assert [row["label"] for row in data["rows"]] == ["Sub-Saharan Africa"]


def test_year_axis_label_is_not_currency_value():
    data = _normalize_chart_data(
        {
            "has_extractable_data": True,
            "chart_type": "line",
            "unit": "Dollars",
            "rows": [
                {
                    "label": "1990",
                    "x": "1990",
                    "value": "1990",
                    "unit": "Dollars",
                    "raw_text": "1990",
                    "evidence_text": "1990",
                }
            ],
            "visible_text": ["1990", "2015", "2016 DOLLARS."],
        },
        "line",
    )

    assert data["has_extractable_data"] is False
    assert data["rows"] == []
