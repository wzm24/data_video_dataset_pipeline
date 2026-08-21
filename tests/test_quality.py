from pathlib import Path

import datavideo.quality as quality
from datavideo.quality import HARD_FAIL, NEEDS_REVIEW, check_clip_quality, run_quality_check
from datavideo.schemas import write_json


def _write_base_clip(root: Path, *, include: bool = True, states: list[dict] | None = None) -> Path:
    clip = root / "bar_1"
    (clip / "keyframes").mkdir(parents=True)
    (clip / "clip.mp4").write_bytes(b"mp4")
    (clip / "keyframes" / "selected.png").write_bytes(b"png")
    (clip / "semantic.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"><g id="entity-car" data-role="entity"><rect id="car-bar" data-role="bar"/></g></svg>\n',
        encoding="utf-8",
    )
    states = states if states is not None else [
        {
            "clip_id": "bar_1",
            "state_id": "state_001",
            "entity_id": "car",
            "metric": "value",
            "value": 20,
            "unit": "%",
            "state_start": 0.0,
            "state_end": 0.0,
            "evidence_frames": [{"frame_id": "f0"}],
        }
    ]
    write_json(
        clip / "dynamic_data.json",
        {
            "include_in_dataset": include,
            "numeric_fact_count": sum(1 for row in states if row.get("value") is not None),
            "dynamic_data": len({row.get("state_id") for row in states}) > 1,
            "data_change_events": [],
            "states": states,
            "exclude_reason": None if include else "no_recoverable_quantitative_data",
        },
    )
    write_json(clip / "chart_data_validation.json", {"has_extractable_data": include})
    write_json(clip / "animation_detection.json", {"is_target_chart_related": True})
    (clip / "dynamic_data.csv").write_text("clip_id,state_id,entity_id,value\nbar_1,state_001,car,20\n", encoding="utf-8")
    (clip / "final_data_table.csv").write_text("clip_id,entity_id,value\nbar_1,car,20\n", encoding="utf-8")
    return clip


def test_quality_check_passes_complete_static_clip(tmp_path):
    clip = _write_base_clip(tmp_path)

    report = check_clip_quality(clip, {"quality": {"enable_vlm": False}})

    assert report["status"] == "pass"
    assert report["issues"] == []


def test_quality_check_flags_missing_required_file(tmp_path):
    clip = _write_base_clip(tmp_path)
    (clip / "semantic.svg").unlink()

    report = check_clip_quality(clip, {"quality": {"enable_vlm": False}})

    assert report["status"] == HARD_FAIL
    assert any(issue["issue_code"] == "missing_required_file" for issue in report["issues"])


def test_quality_check_accepts_standard_no_data_exclusion(tmp_path):
    clip = _write_base_clip(tmp_path, include=False, states=[])
    (clip / "dynamic_data.csv").unlink()
    (clip / "final_data_table.csv").unlink()

    report = check_clip_quality(clip, {"quality": {"enable_vlm": False}})

    assert not any(issue["issue_code"] == "included_without_numeric_facts" for issue in report["issues"])


def test_quality_check_accepts_multi_state_without_semantic_state_inputs(tmp_path):
    clip = _write_base_clip(
        tmp_path,
        states=[
            {"state_id": "state_001", "entity_id": "car", "metric": "value", "value": 10, "state_start": 0.0, "state_end": 0.0, "evidence_frames": [{"frame_id": "f0", "time_seconds": 0.0, "path": "f0.png"}]},
            {"state_id": "state_002", "entity_id": "car", "metric": "value", "value": 20, "state_start": 1.0, "state_end": 1.0, "evidence_frames": [{"frame_id": "f1", "time_seconds": 1.0, "path": "f1.png"}]},
        ],
    )

    report = check_clip_quality(clip, {"quality": {"enable_vlm": False}})

    codes = {issue["issue_code"] for issue in report["issues"]}
    # Per-state semantic inputs/SVGs are no longer required: each state is
    # an independent data-driven SVG generation and can be adjusted manually.
    assert "missing_semantic_state_inputs" not in codes
    assert "missing_semantic_state_svgs" not in codes


def test_quality_check_flags_data_entity_missing_from_semantic(tmp_path):
    clip = _write_base_clip(
        tmp_path,
        states=[
            {"state_id": "state_001", "entity_id": "plane", "metric": "value", "value": 10, "evidence_frames": [{"frame_id": "f0"}]},
        ],
    )

    report = check_clip_quality(clip, {"quality": {"enable_vlm": False}})

    assert any(issue["issue_code"] == "data_entity_missing_in_semantic" for issue in report["issues"])


def test_quality_check_writes_review_queue(tmp_path):
    generated = tmp_path / "generated"
    clip = _write_base_clip(generated)
    (clip / "semantic.svg").unlink()

    report = run_quality_check({"generated_root": str(generated), "quality": {"enable_vlm": False}})

    assert report["flagged_clip_count"] == 1
    assert (generated / "quality" / "quality_report.json").exists()
    assert (generated / "quality" / "quality_flags.csv").exists()
    assert (generated / "quality" / "quality_review_queue.csv").exists()


def test_quality_check_uses_separate_vlm_interface(monkeypatch, tmp_path):
    clip = _write_base_clip(tmp_path)

    class FakeQualityClient:
        def review_quality(self, image_paths, prompt):
            assert image_paths
            assert "Artifact summaries" in prompt
            return {
                "result": {
                    "needs_review": True,
                    "severity": "medium",
                    "issue_codes": ["fake_visual_mismatch"],
                    "evidence": ["semantic overlay misses a visible bar"],
                    "recommended_action": "manual_review",
                }
            }

    monkeypatch.setattr(quality, "make_quality_client", lambda cfg: FakeQualityClient())

    report = check_clip_quality(clip, {"quality": {"enable_vlm": True, "model": {"client_module": "fake"}}})

    assert any(issue["layer"] == "vlm" and issue["issue_code"] == "fake_visual_mismatch" for issue in report["issues"])
