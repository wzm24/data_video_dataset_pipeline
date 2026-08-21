from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .model_client import make_quality_client
from .schemas import ensure_dir, read_json, write_csv, write_json


HARD_FAIL = "hard_fail"
NEEDS_REVIEW = "needs_review"
PASS = "pass"


def _clip_id(path: Path) -> str:
    return path.name


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _issue(
    clip_id: str,
    severity: str,
    issue_code: str,
    reason: str,
    artifact: str,
    *,
    layer: str,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "severity": severity,
        "issue_code": issue_code,
        "reason": reason,
        "artifact": artifact,
        "layer": layer,
    }


def _svg_roles(svg_path: Path) -> dict[str, set[str]]:
    roles: dict[str, set[str]] = {"ids": set(), "entity_ids": set(), "roles": set()}
    if not svg_path.exists():
        return roles
    root = ET.parse(svg_path).getroot()
    for elem in root.iter():
        elem_id = elem.attrib.get("id")
        if elem_id:
            roles["ids"].add(elem_id)
            if elem_id.startswith("entity-"):
                roles["entity_ids"].add(elem_id.removeprefix("entity-"))
        role = elem.attrib.get("data-role")
        if role:
            roles["roles"].add(role)
        entity_id = elem.attrib.get("data-entity-id")
        if entity_id:
            roles["entity_ids"].add(entity_id)
    return roles


def _json_or_issue(path: Path, clip_id: str, issues: list[dict[str, Any]], artifact_name: str) -> Any | None:
    if not path.exists():
        issues.append(_issue(clip_id, HARD_FAIL, "missing_required_file", f"Missing {artifact_name}", str(path), layer="rules"))
        return None
    try:
        return read_json(path)
    except Exception as exc:
        issues.append(_issue(clip_id, HARD_FAIL, "invalid_json", f"Could not parse {artifact_name}: {exc}", str(path), layer="rules"))
        return None


def _state_entity_sets(states: list[dict[str, Any]]) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = {}
    for row in states:
        state_id = str(row.get("state_id") or "")
        entity_id = str(row.get("entity_id") or "")
        if state_id and entity_id:
            groups.setdefault(state_id, set()).add(entity_id)
    return groups


def _rule_check_clip(clip_root: Path) -> list[dict[str, Any]]:
    clip_id = _clip_id(clip_root)
    issues: list[dict[str, Any]] = []
    required = [
        "clip.mp4",
        "keyframes/selected.png",
        "semantic.svg",
        "dynamic_data.json",
        "chart_data_validation.json",
        "animation_detection.json",
    ]
    for rel in required:
        if not (clip_root / rel).exists():
            issues.append(_issue(clip_id, HARD_FAIL, "missing_required_file", f"Missing {rel}", str(clip_root / rel), layer="rules"))

    dynamic = _json_or_issue(clip_root / "dynamic_data.json", clip_id, issues, "dynamic_data.json")
    validation = _json_or_issue(clip_root / "chart_data_validation.json", clip_id, issues, "chart_data_validation.json")
    if not isinstance(dynamic, dict):
        return issues

    states = dynamic.get("states") if isinstance(dynamic.get("states"), list) else []
    events = dynamic.get("data_change_events") if isinstance(dynamic.get("data_change_events"), list) else []
    include = bool(dynamic.get("include_in_dataset"))
    numeric_count = int(dynamic.get("numeric_fact_count") or 0)
    exclude_reason = dynamic.get("exclude_reason")

    if include and numeric_count <= 0:
        issues.append(_issue(clip_id, HARD_FAIL, "included_without_numeric_facts", "Dataset-included sample has no numeric facts", str(clip_root / "dynamic_data.json"), layer="rules"))
    if not include and exclude_reason != "no_recoverable_quantitative_data":
        issues.append(_issue(clip_id, NEEDS_REVIEW, "excluded_without_standard_reason", "Excluded sample does not use the standard no-data reason", str(clip_root / "dynamic_data.json"), layer="rules"))
    if include and not states:
        issues.append(_issue(clip_id, HARD_FAIL, "included_without_states", "Included sample has no dynamic/static data states", str(clip_root / "dynamic_data.json"), layer="rules"))
    if dynamic.get("dynamic_data") and len({row.get("state_id") for row in states}) < 2 and not events:
        issues.append(_issue(clip_id, NEEDS_REVIEW, "dynamic_flag_without_multiple_states_or_events", "dynamic_data=true but no multiple states/events were found", str(clip_root / "dynamic_data.json"), layer="rules"))
    if validation and bool(validation.get("has_extractable_data")) != include:
        issues.append(_issue(clip_id, NEEDS_REVIEW, "validation_include_mismatch", "chart_data_validation disagrees with dynamic_data include flag", str(clip_root / "chart_data_validation.json"), layer="rules"))

    dynamic_csv = _read_csv(clip_root / "dynamic_data.csv")
    if include and not dynamic_csv:
        issues.append(_issue(clip_id, HARD_FAIL, "missing_dynamic_data_csv", "Included sample is missing dynamic_data.csv rows", str(clip_root / "dynamic_data.csv"), layer="rules"))
    final_csv = _read_csv(clip_root / "final_data_table.csv")
    if include and not final_csv:
        issues.append(_issue(clip_id, HARD_FAIL, "missing_final_data_table", "Included sample is missing final_data_table.csv rows", str(clip_root / "final_data_table.csv"), layer="rules"))
    events_csv = _read_csv(clip_root / "data_change_events.csv")
    if events and len(events_csv) != len(events):
        issues.append(_issue(clip_id, NEEDS_REVIEW, "event_csv_json_count_mismatch", "data_change_events.csv row count differs from dynamic_data.json", str(clip_root / "data_change_events.csv"), layer="rules"))

    return issues


def _cross_artifact_check_clip(clip_root: Path) -> list[dict[str, Any]]:
    clip_id = _clip_id(clip_root)
    issues: list[dict[str, Any]] = []
    dynamic_path = clip_root / "dynamic_data.json"
    if not dynamic_path.exists():
        return issues
    try:
        dynamic = read_json(dynamic_path)
    except Exception:
        return issues
    states = dynamic.get("states") if isinstance(dynamic.get("states"), list) else []
    if not states:
        return issues

    data_entities = {str(row.get("entity_id")) for row in states if row.get("entity_id")}
    top_svg = clip_root / "semantic.svg"
    if top_svg.exists():
        try:
            roles = _svg_roles(top_svg)
            if "bar" not in roles["roles"] and any(str(row.get("metric")) != "value" for row in states):
                issues.append(_issue(clip_id, NEEDS_REVIEW, "semantic_missing_bar_role", "semantic.svg has no bar role for bar-like recovered data", str(top_svg), layer="cross_artifact"))
            missing = sorted(entity for entity in data_entities if entity not in roles["entity_ids"] and f"entity-{entity}" not in roles["ids"])
            if missing:
                issues.append(_issue(clip_id, NEEDS_REVIEW, "data_entity_missing_in_semantic", f"Recovered entities missing from semantic.svg: {', '.join(missing)}", str(top_svg), layer="cross_artifact"))
        except Exception as exc:
            issues.append(_issue(clip_id, HARD_FAIL, "invalid_semantic_svg", f"Could not parse semantic.svg: {exc}", str(top_svg), layer="cross_artifact"))

    state_manifest_path = clip_root / "semantic_state_svg_manifest.json"
    if state_manifest_path.exists():
        try:
            state_manifest = read_json(state_manifest_path)
        except Exception:
            return issues
        data_by_state: dict[str, set[str]] = {}
        for row in states:
            state_id = str(row.get("state_id") or "")
            if state_id:
                data_by_state.setdefault(state_id, set()).add(str(row.get("entity_id")))
        for item in state_manifest.get("semantic_svgs", []) if isinstance(state_manifest.get("semantic_svgs"), list) else []:
            state_id = str(item.get("state_id") or "")
            svg_path = Path(item.get("semantic_svg") or "")
            if not svg_path.exists():
                issues.append(_issue(clip_id, NEEDS_REVIEW, "missing_state_semantic_svg", f"Missing semantic SVG for {state_id}", str(svg_path), layer="cross_artifact"))
                continue
            try:
                roles = _svg_roles(svg_path)
            except Exception as exc:
                issues.append(_issue(clip_id, HARD_FAIL, "invalid_state_semantic_svg", f"Could not parse state semantic SVG: {exc}", str(svg_path), layer="cross_artifact"))
                continue
            missing = sorted(entity for entity in data_by_state.get(state_id, set()) if entity not in roles["entity_ids"] and f"entity-{entity}" not in roles["ids"])
            if missing:
                issues.append(_issue(clip_id, NEEDS_REVIEW, "state_data_entity_missing_in_semantic", f"{state_id} entities missing from state semantic SVG: {', '.join(missing)}", str(svg_path), layer="cross_artifact"))
    return issues


def _quality_vlm_prompt(clip_id: str, summaries: dict[str, Any]) -> str:
    return (
        "You are auditing generated data-video dataset artifacts. "
        "Return strict JSON with keys needs_review, severity, issue_codes, evidence, recommended_action. "
        "Flag likely visual/data/semantic mismatches, but do not approve samples solely from confidence.\n"
        f"Clip id: {clip_id}\n"
        f"Artifact summaries:\n{summaries}"
    )


def _vlm_check_clip(clip_root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    clip_id = _clip_id(clip_root)
    qc_cfg = cfg.get("quality", {})
    if not qc_cfg.get("enable_vlm", False):
        return []
    try:
        client = make_quality_client(cfg)
    except Exception as exc:
        return [_issue(clip_id, NEEDS_REVIEW, "qc_vlm_unavailable", f"Quality model unavailable: {exc}", str(clip_root), layer="vlm")]
    if not hasattr(client, "review_quality"):
        return [_issue(clip_id, NEEDS_REVIEW, "qc_vlm_interface_missing", "Quality model client does not implement review_quality", str(clip_root), layer="vlm")]

    images = []
    for rel in ["keyframes/selected.png", "semantic_preview.png"]:
        path = clip_root / rel
        if path.exists():
            images.append(str(path))
    for state_dir in sorted((clip_root / "semantic_states").glob("*/")):
        for rel in ["semantic_preview.png", "vision/semantic_preview.png"]:
            path = state_dir / rel
            if path.exists():
                images.append(str(path))
    for state_keyframe in sorted((clip_root / "keyframes" / "states").glob("state_*.png")):
        images.append(str(state_keyframe))
    summaries = {}
    for rel in ["dynamic_data.json", "animation_detection.json", "semantic_scene.json", "semantic_state_svg_manifest.json"]:
        path = clip_root / rel
        if path.exists():
            try:
                summaries[rel] = read_json(path)
            except Exception:
                summaries[rel] = {"parse_error": True}
    try:
        response = client.review_quality(images, _quality_vlm_prompt(clip_id, summaries))
    except Exception as exc:
        return [_issue(clip_id, NEEDS_REVIEW, "qc_vlm_failed", f"Quality model failed: {exc}", str(clip_root), layer="vlm")]
    result = response.get("result") if isinstance(response, dict) else None
    if not isinstance(result, dict):
        return [_issue(clip_id, NEEDS_REVIEW, "qc_vlm_bad_response", "Quality model returned no structured result", str(clip_root), layer="vlm")]
    if not result.get("needs_review"):
        return []
    codes = result.get("issue_codes") if isinstance(result.get("issue_codes"), list) else ["qc_vlm_flagged"]
    reason = "; ".join(str(item) for item in result.get("evidence", [])) or str(result.get("recommended_action") or "Quality model flagged this sample")
    return [
        _issue(
            clip_id,
            str(result.get("severity") or NEEDS_REVIEW),
            str(code),
            reason,
            str(clip_root),
            layer="vlm",
        )
        for code in codes
    ]


def check_clip_quality(clip_root: str | Path, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    clip_root = Path(clip_root)
    cfg = cfg or {}
    issues = [
        *_rule_check_clip(clip_root),
        *_cross_artifact_check_clip(clip_root),
        *_vlm_check_clip(clip_root, cfg),
    ]
    status = PASS
    if any(issue["severity"] == HARD_FAIL for issue in issues):
        status = HARD_FAIL
    elif issues:
        status = NEEDS_REVIEW
    return {
        "clip_id": _clip_id(clip_root),
        "clip_root": str(clip_root),
        "status": status,
        "needs_review": bool(issues),
        "issues": issues,
    }


def run_quality_check(cfg: dict[str, Any], force: bool = False) -> dict[str, Any]:
    generated_root = Path(cfg.get("generated_root", "data/generated"))
    quality_dir = ensure_dir(generated_root / "quality")
    clip_id = cfg.get("clip_id")
    clip_roots = [generated_root / clip_id] if clip_id else [path for path in sorted(generated_root.iterdir()) if path.is_dir() and path.name != "quality"]
    reports = [check_clip_quality(path, cfg) for path in clip_roots if path.exists()]
    flags = [issue for report in reports for issue in report["issues"]]
    queue = [
        {
            "clip_id": report["clip_id"],
            "status": report["status"],
            "issue_count": len(report["issues"]),
            "issue_codes": "|".join(issue["issue_code"] for issue in report["issues"]),
            "review_target": report["issues"][0]["artifact"] if report["issues"] else report["clip_root"],
        }
        for report in reports
        if report["needs_review"]
    ]
    summary = {
        "clip_count": len(reports),
        "flagged_clip_count": len(queue),
        "hard_fail_clip_count": sum(1 for report in reports if report["status"] == HARD_FAIL),
        "needs_review_clip_count": sum(1 for report in reports if report["status"] == NEEDS_REVIEW),
        "quality_report": str(quality_dir / "quality_report.json"),
        "quality_flags_csv": str(quality_dir / "quality_flags.csv") if flags else None,
        "quality_review_queue_csv": str(quality_dir / "quality_review_queue.csv") if queue else None,
        "clips": reports,
    }
    write_json(quality_dir / "quality_report.json", summary)
    if flags:
        write_csv(quality_dir / "quality_flags.csv", flags)
    if queue:
        write_csv(quality_dir / "quality_review_queue.csv", queue)
    return summary
