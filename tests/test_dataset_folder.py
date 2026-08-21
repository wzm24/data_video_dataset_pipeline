import json

from datavideo.multichart_pipeline import _line_metadata_from_dynamic, _series_label_from_title, build_dataset_folder


def test_series_label_from_title_strips_location_qualifier():
    assert _series_label_from_title("'Net additions' in England") == "Net additions"
    assert _series_label_from_title("Productivity") == "Productivity"
    assert _series_label_from_title("") == ""


def test_line_metadata_from_dynamic_preserves_unit_and_x_labels():
    metadata = _line_metadata_from_dynamic(
        {
            "states": [
                {"entity": "GM", "state_key": "2012", "value": 80.0, "unit": "%"},
                {"entity": "GM", "state_key": "2013", "value": 50.0, "unit": "%"},
            ]
        },
        title="Share of profits spent on stock buybacks",
    )

    assert metadata["unit"] == "%"
    assert metadata["x_labels"] == ["2012", "2013"]
    assert metadata["series"] == [{"name": "GM", "values": [80.0, 50.0], "x_labels": ["2012", "2013"]}]


def _make_clip(tmp_path):
    clip = tmp_path / "combined_1"
    for state_dir in ["1990", "2017"]:
        (clip / "semantic_states" / state_dir).mkdir(parents=True)
        (clip / "semantic_states" / state_dir / "semantic.svg").write_text(f"<svg>{state_dir}</svg>", encoding="utf-8")
        (clip / "semantic_states" / state_dir / "semantic_components.svg").write_text(f"<svg>{state_dir}-comp</svg>", encoding="utf-8")
    # stale old-naming dir that must NOT be referenced
    (clip / "semantic_states" / "state_001_1990").mkdir(parents=True)
    (clip / "semantic_states" / "state_001_1990" / "semantic.svg").write_text("old", encoding="utf-8")
    (clip / "keyframes" / "states").mkdir(parents=True)
    (clip / "keyframes" / "states" / "state_001_1990.png").write_bytes(b"kf1990")
    (clip / "keyframes" / "states" / "state_002_2017.png").write_bytes(b"kf2017")
    (clip / "keyframes" / "selected.png").write_bytes(b"selected")
    (clip / "aligned_overlay.png").write_bytes(b"overlay")

    dynamic = {
        "states": [
            {
                "state_id": "state_001",
                "state_key": "1990",
                "state_label": "1990",
                "entity_id": "a",
                "entity": "A",
                "metric": "Rate",
                "value": 48.0,
                "unit": "%",
                "value_type": "exact",
                "source_type": "visual",
                "confidence": 0.8,
                "review_status": "machine",
                "state_start": 0.0,
                "state_end": 0.75,
            },
            {
                "state_id": "state_002",
                "state_key": "2017",
                "state_label": "2017",
                "entity_id": "a",
                "entity": "A",
                "metric": "Rate",
                "value": 36.1,
                "unit": "%",
                "value_type": "exact",
                "source_type": "visual_frame_align",
                "confidence": 0.85,
                "review_status": "machine",
                "state_start": 3.75,
                "state_end": 3.75,
            },
        ]
    }
    (clip / "dynamic_data.json").write_text(json.dumps(dynamic), encoding="utf-8")
    (clip / "dynamic_data.csv").write_text(
        "clip_id,state_key,state_label,entity_id,entity,metric,value,unit,value_type,source_type,confidence,review_status,state_start,state_end\n"
        "combined_1,1990,1990,a,A,Rate,48.0,%,exact,visual,0.8,machine,0.0,0.75\n"
        "combined_1,2017,2017,a,A,Rate,36.1,%,exact,visual_frame_align,0.85,machine,3.75,3.75\n",
        encoding="utf-8",
    )
    (clip / "final_data_table.csv").write_text(
        "clip_id,entity_id,entity,metric,value,unit,type,source_type,confidence,review_status\n"
        "combined_1,a,A,Rate,36.1,%,exact,visual_frame_align,0.85,machine\n",
        encoding="utf-8",
    )
    animation = {
        "target_chart_type": "bar",
        "overall_description": "bars grow",
        "major_actions": [{"action": "bar_grow", "description": "grows", "evidence_timestamps": [0.0, 3.75]}],
        "confidence": 0.9,
        "model_status": "qwen",
        "prompt_version": "x_animation_v6",
    }
    (clip / "animation_detection.json").write_text(json.dumps(animation), encoding="utf-8")
    (clip / "keyframes" / "keyframe_manifest.json").write_text(
        json.dumps(
            {
                "states": [
                    {"state_key": "1990", "asset": str(clip / "keyframes" / "states" / "state_001_1990.png")},
                    {"state_key": "2017", "asset": str(clip / "keyframes" / "states" / "state_002_2017.png")},
                ]
            }
        ),
        encoding="utf-8",
    )
    clip_report = {
        "clip": {
            "clip_id": "combined_1",
            "raw_video_title": "T",
            "chart_type": "bar",
            "start_seconds": 0.0,
            "end_seconds": 4.0,
        },
        "context": {},
        "asr": {},
        "keyframes": {
            "timestamps": {"selected": 3.75},
            "assets": {"selected": str(clip / "keyframes" / "selected.png")},
        },
    }
    return clip, clip_report


def test_build_dataset_folder_creates_per_state_subfolders(tmp_path):
    clip, clip_report = _make_clip(tmp_path)

    result = build_dataset_folder(clip, clip_report)

    dataset = clip / "dataset"
    assert result["state_count"] == 2
    assert (dataset / "data_table.csv").exists()
    assert (dataset / "intent.json").exists()
    assert (dataset / "keyframe.png").exists()
    assert (dataset / "aligned_overlay.png").exists()
    assert (dataset / "manifest.json").exists()
    # primary state follows the selected keyframe timestamp (2017)
    assert "<svg>2017</svg>" in (dataset / "semantic.svg").read_text(encoding="utf-8")
    for state_key in ["1990", "2017"]:
        state_dir = dataset / "states" / state_key
        assert (state_dir / "semantic.svg").exists()
        assert (state_dir / "data_table.csv").exists()
        assert (state_dir / "intent.json").exists()
        static_intent = json.loads((state_dir / "intent.json").read_text(encoding="utf-8"))
        assert static_intent["is_static"] is True
        assert static_intent["state_key"] == state_key
    # stale old-naming dirs are never referenced
    assert not (dataset / "states" / "state_001_1990").exists()
    # intent was reconciled against data: grow -> shrink
    intent = json.loads((dataset / "intent.json").read_text(encoding="utf-8"))
    assert intent["reconciled_with_data"] is True
    assert intent["major_actions"][0]["action"] == "bar_shrink"
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_state_count"] == 2
    assert {state["state_key"] for state in manifest["states"]} == {"1990", "2017"}


def test_build_dataset_folder_static_clip_keeps_flat_layout(tmp_path):
    clip = tmp_path / "bar_1"
    (clip / "keyframes").mkdir(parents=True)
    (clip / "keyframes" / "selected.png").write_bytes(b"selected")
    (clip / "semantic.svg").write_text("<svg>static</svg>", encoding="utf-8")
    (clip / "semantic_components.svg").write_text("<svg>static-comp</svg>", encoding="utf-8")
    (clip / "final_data_table.csv").write_text(
        "clip_id,entity_id,entity,metric,value,unit,type,source_type,confidence,review_status\n"
        "bar_1,a,A,Rate,10.0,%,exact,visual_frame_align,0.85,machine\n",
        encoding="utf-8",
    )
    (clip / "animation_detection.json").write_text(
        json.dumps({"target_chart_type": "bar", "overall_description": "static", "major_actions": [], "confidence": 0.5}),
        encoding="utf-8",
    )
    clip_report = {
        "clip": {"clip_id": "bar_1", "raw_video_title": "T", "chart_type": "bar", "start_seconds": 0.0, "end_seconds": 2.0},
        "context": {},
        "asr": {},
        "keyframes": {"timestamps": {"selected": 1.0}, "assets": {"selected": str(clip / "keyframes" / "selected.png")}},
    }

    result = build_dataset_folder(clip, clip_report)

    dataset = clip / "dataset"
    assert result["state_count"] == 0
    assert (dataset / "semantic.svg").exists()
    assert (dataset / "data_table.csv").exists()
    assert (dataset / "intent.json").exists()
    assert not (dataset / "states").exists()


def test_build_dataset_folder_uses_plateau_states_when_scan_exists(tmp_path):
    clip = tmp_path / "bar_74"
    for state_id in ["state_01", "state_02"]:
        state_dir = clip / "semantic_states" / state_id
        state_dir.mkdir(parents=True)
        (state_dir / "semantic.svg").write_text(f"<svg>{state_id}</svg>", encoding="utf-8")
        (state_dir / "data_table.csv").write_text("entity,value\nA,10\n", encoding="utf-8")
        (state_dir / "intent.json").write_text(json.dumps({"state_key": state_id}), encoding="utf-8")
    (clip / "state_scan_report.json").write_text(
        json.dumps(
            {
                "states": [
                    {"state_id": "state_01", "start": 0.0, "end": 2.0, "bar_count": 4},
                    {"state_id": "state_02", "start": 5.0, "end": 8.0, "bar_count": 4},
                ]
            }
        ),
        encoding="utf-8",
    )
    (clip / "keyframes").mkdir(parents=True)
    (clip / "keyframes" / "selected.png").write_bytes(b"selected")
    (clip / "semantic.svg").write_text("<svg>primary</svg>", encoding="utf-8")
    (clip / "final_data_table.csv").write_text("entity,value\nA,20\n", encoding="utf-8")
    (clip / "animation_detection.json").write_text(
        json.dumps({"major_actions": [], "overall_description": "x"}),
        encoding="utf-8",
    )
    clip_report = {
        "clip": {"clip_id": "bar_74", "raw_video_title": "T", "chart_type": "bar", "start_seconds": 0.0, "end_seconds": 10.0},
        "context": {},
        "asr": {},
        "keyframes": {"timestamps": {"selected": 9.0}, "assets": {"selected": str(clip / "keyframes" / "selected.png")}},
    }

    result = build_dataset_folder(clip, clip_report)

    dataset = clip / "dataset"
    assert result["state_count"] == 2
    for state_key in ["state-01", "state-02"]:
        state_dir = dataset / "states" / state_key
        assert (state_dir / "semantic.svg").exists()
        assert (state_dir / "data_table.csv").exists()
        assert (state_dir / "intent.json").exists()
    # The primary semantic.svg still comes from the clip's final render.
    assert (dataset / "semantic.svg").read_text(encoding="utf-8") == "<svg>primary</svg>"
