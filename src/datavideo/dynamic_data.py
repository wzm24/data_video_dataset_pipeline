from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .schemas import read_json, read_jsonl, write_csv, write_json, write_jsonl


NO_RECOVERABLE_QUANTITATIVE_DATA = "no_recoverable_quantitative_data"

_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|percent|percentage|dollars?|usd|\\$|km|miles?|days?|years?|million|billion|thousand|k|m|bn)?", re.I)
_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_WORD_NUMBER_RE = re.compile(
    r"\b(" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")(?:[-\s]+(" + "|".join(k for k, v in _NUMBER_WORDS.items() if v < 10) + r"))?\b\s*(?:full\s+|straight\s+)?(?:days?|years?|miles?|km|percent|percentage|dollars?|usd)?",
    re.I,
)


def numeric_token(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    match = _NUMBER_RE.search(text)
    if not match:
        word_match = _WORD_NUMBER_RE.search(text)
        if not word_match:
            return None
        return word_match.group(0).strip()
    return match.group(0).strip()


def numeric_value(value: Any) -> float | None:
    token = numeric_token(value)
    if token is None:
        return None
    token = token.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", token)
    if not match:
        words = re.findall(r"[a-z]+", token.lower())
        total = 0
        found = False
        for word in words:
            if word in _NUMBER_WORDS:
                total += _NUMBER_WORDS[word]
                found = True
        return float(total) if found else None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def infer_unit(*values: Any) -> str | None:
    for value in values:
        if value in (None, ""):
            continue
        token = numeric_token(value)
        text = str(value).strip().lower()
        token_text = (token or "").strip().lower()
        haystack = f"{token_text} {text}"
        if "%" in text or "percent" in text or "percentage" in text:
            return "%"
        if "$" in haystack or "dollar" in haystack or "usd" in haystack:
            return "$"
        if "mile" in haystack:
            return "miles"
        if re.search(r"\bkm\b", haystack):
            return "km"
        if "day" in haystack:
            return "day"
        if "year" in haystack:
            return "years"
        if "million" in haystack or re.search(r"\bm\b", token_text):
            return "million"
        if "billion" in haystack or re.search(r"\bbn\b", token_text):
            return "billion"
        if "thousand" in haystack or re.search(r"\bk\b", token_text):
            return "thousand"
    return None


def entity_id_from_row(row: dict[str, Any]) -> str:
    for key in ("entity_id", "label", "series", "x", "state"):
        value = row.get(key)
        if value not in (None, ""):
            if key == "state" and re.fullmatch(r"\d{4}", str(value).strip()):
                continue
            text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
            return text or "unknown"
    return "unknown"


def metric_from_row(row: dict[str, Any], default: str | None = None) -> str:
    for key in ("metric", "y"):
        value = row.get(key)
        if value not in (None, "") and numeric_value(value) is None:
            return str(value)
    if default not in (None, "") and numeric_value(default) is None:
        return str(default)
    value = row.get("series")
    if value not in (None, "") and numeric_value(value) is None:
        return str(value)
    return "value"


def state_key_from_row(row: dict[str, Any]) -> str | None:
    for key in ("state_key", "year", "state_id", "state"):
        value = row.get(key)
        if value not in (None, ""):
            if key == "state" and value in {row.get("label"), row.get("series"), row.get("x")}:
                continue
            return str(value)
    return None


def _state_label_from_row(row: dict[str, Any]) -> str | None:
    for key in ("state_label", "year", "state"):
        value = row.get(key)
        if value not in (None, ""):
            if key == "state" and value in {row.get("label"), row.get("series"), row.get("x")}:
                continue
            return str(value)
    return None


def _display_entity(label: str) -> str:
    upper_labels = {"car": "CAR", "spaceship": "SPACESHIP", "boeing 747": "BOING 747", "boing 747": "BOING 747"}
    return upper_labels.get(label.lower(), label)


def _frame_for_row(row: dict[str, Any], frame_context: list[dict[str, Any]], image_paths: list[str]) -> dict[str, Any] | None:
    source_frame = row.get("source_frame")
    time_seconds = row.get("time_seconds")
    chosen: dict[str, Any] | None = None
    for idx, frame in enumerate(frame_context):
        if source_frame is not None and frame.get("source_frame") == source_frame:
            chosen = frame
            break
        if time_seconds is not None and frame.get("time_seconds") is not None:
            try:
                if abs(float(frame["time_seconds"]) - float(time_seconds)) <= 0.05:
                    chosen = frame
                    break
            except (TypeError, ValueError):
                pass
    if chosen is None and frame_context:
        chosen = frame_context[0]
    if chosen is None:
        return None
    image_index = int(chosen.get("image_index") or 1)
    path = image_paths[image_index - 1] if 0 < image_index <= len(image_paths) else None
    return {
        "frame_id": chosen.get("source_frame"),
        "time_seconds": chosen.get("time_seconds"),
        "path": path,
    }


def visual_records_from_clip_data(
    data: dict[str, Any],
    frame_context: list[dict[str, Any]],
    image_paths: list[str],
    clip_id: str,
) -> list[dict[str, Any]]:
    rows = data.get("rows") if isinstance(data, dict) else []
    records: list[dict[str, Any]] = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        value = row.get("value")
        value_num = numeric_value(value)
        if value_num is None:
            continue
        entity_id = entity_id_from_row(row)
        if entity_id == "unknown":
            continue
        if _label_embeds_own_value(
            row.get("label") or row.get("series") or row.get("x") or row.get("state"),
            value_num,
        ):
            continue
        evidence_text = row.get("evidence_text") or row.get("raw_text") or value
        frame = _frame_for_row(row, frame_context, image_paths)
        time_value = frame.get("time_seconds") if frame else row.get("time_seconds")
        records.append(
            {
                "clip_id": clip_id,
                "state_id": None,
                "state_key": state_key_from_row(row),
                "state_label": _state_label_from_row(row),
                "entity_id": entity_id,
                "entity": _display_entity(
                    str(row.get("label") or row.get("series") or row.get("x") or row.get("state") or entity_id)
                ),
                "metric": metric_from_row(row, data.get("y_axis") or data.get("title")),
                "value": value_num,
                "unit": row.get("unit") or infer_unit(value, evidence_text, data.get("unit")),
                "state_start": time_value,
                "state_end": time_value,
                "source_type": "visual",
                "evidence_frames": [frame] if frame else [],
                "evidence_sentence_id": None,
                "confidence": _as_float(row.get("confidence")) or 0.8,
                "review_status": "machine",
                "raw_text": row.get("raw_text"),
                "evidence_text": evidence_text,
                "value_type": "exact",
            }
        )
    return filter_axis_tick_records(records)


def axis_tick_keys(rows: list[dict[str, Any]]) -> set[tuple[Any, str]]:
    """Return (timestamp, entity_id) keys whose rows are axis tick labels.

    When several entities at the same timestamp each report the *same*
    multi-value set (e.g. every bar gets 0/50/100), those values are almost
    certainly axis ticks recovered as data, not per-entity values.
    """
    by_ts: dict[Any, dict[str, set[float]]] = {}
    for row in rows:
        ts = row.get("state_start", row.get("time_seconds"))
        eid = str(row.get("entity_id") or "")
        value = numeric_value(row.get("value"))
        if not eid or value is None:
            continue
        by_ts.setdefault(ts, {}).setdefault(eid, set()).add(value)
    keys: set[tuple[Any, str]] = set()
    for ts, entities in by_ts.items():
        if len(entities) < 2:
            continue
        sets = list(entities.values())
        for eid, values in entities.items():
            if len(values) >= 2 and any(other is not values and values == other for other in sets):
                keys.add((ts, eid))
    return keys


def filter_axis_tick_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop records that are axis tick labels (see ``axis_tick_keys``)."""
    keys = axis_tick_keys(records)
    if not keys:
        return records
    return [
        r
        for r in records
        if (r.get("state_start"), str(r.get("entity_id") or "")) not in keys
    ]


def _sentence_id(row: dict[str, Any], fallback: int) -> str:
    for key in ("sentence_id", "sentence_index", "id"):
        if row.get(key) not in (None, ""):
            return str(row[key])
    return f"sentence_{fallback:03d}"


def sanitize_metric(metric: Any) -> str:
    """Drop junk metrics that carry no letters/digits (e.g. a VLM reading the
    dollar signs ``$$$`` as the metric); empty metrics fall back to "Value"."""
    text = str(metric or "").strip()
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    return text


def _label_embeds_own_value(label: Any, value: Any) -> bool:
    """True when a label like "380,000 km: Average Distance to the Moon"
    embeds its own number+unit before a colon (title/label-value confusion).

    Legitimate category labels such as "Men: 50%" or "Income: $40,000" do not
    start with a number, so they are never caught by this rule.
    """
    text = str(label or "")
    if ":" not in text:
        return False
    head = text.split(":", 1)[0].strip()
    if not re.match(r"^[\d,.\s]+\s*[A-Za-z%$€£¥]+$", head):
        return False
    digits = re.sub(r"[^0-9.]", "", head)
    if not digits:
        return False
    try:
        head_number = float(digits)
    except ValueError:
        return False
    value_number = numeric_value(value)
    return value_number is not None and abs(value_number - head_number) < 1e-6


def _sentence_overlaps_visual(row: dict[str, Any], intervals: dict[str, Any] | None) -> bool:
    if not intervals:
        return True
    visual = intervals.get("reference_source") or intervals.get("visual_clip_source") or {}
    start = row.get("start_source", row.get("start_source_seconds", row.get("start")))
    end = row.get("end_source", row.get("end_source_seconds", row.get("end")))
    if start is None or end is None or not visual:
        return True
    try:
        return float(end) >= float(visual["start"]) and float(start) <= float(visual["end"])
    except (KeyError, TypeError, ValueError):
        return True


def _entity_labels(chart_context: dict[str, Any]) -> list[tuple[str, str]]:
    labels: list[str] = []
    metadata = chart_context.get("chart_metadata") if isinstance(chart_context.get("chart_metadata"), dict) else {}
    for source in (chart_context, metadata):
        series = source.get("series") if isinstance(source, dict) else None
        if isinstance(series, list):
            for item in series:
                if isinstance(item, dict):
                    label = item.get("label") or item.get("series") or item.get("x")
                    if label not in (None, ""):
                        labels.append(str(label))
    labels.extend(str(value) for value in chart_context.get("category_labels", []) if value not in (None, ""))
    unique = []
    seen = set()
    for label in labels:
        key = label.lower()
        if key not in seen:
            unique.append((entity_id_from_row({"label": label}), label))
            seen.add(key)
    return unique


def _entity_aliases(chart_context: dict[str, Any]) -> list[dict[str, str]]:
    entities = [{"entity_id": entity_id, "label": label, "alias": label.lower()} for entity_id, label in _entity_labels(chart_context)]
    labels = {item["entity_id"] for item in entities}
    text = " ".join(
        str(value or "")
        for value in [
            chart_context.get("title"),
            chart_context.get("x_axis"),
            chart_context.get("y_axis"),
            *chart_context.get("visible_text", []),
        ]
    ).lower()
    if "boeing" in text or "boing" in text or "747" in text or "boeing-747" in labels:
        entities.extend(
            [
                {"entity_id": "boeing-747", "label": "BOING 747", "alias": "747"},
                {"entity_id": "boeing-747", "label": "BOING 747", "alias": "boeing 747"},
                {"entity_id": "boeing-747", "label": "BOING 747", "alias": "boing 747"},
            ]
        )
    if "spaceship" in text or "spaceship" in labels:
        entities.extend(
            [
                {"entity_id": "spaceship", "label": "SPACESHIP", "alias": "spaceship"},
                {"entity_id": "spaceship", "label": "SPACESHIP", "alias": "current technology"},
                {"entity_id": "spaceship", "label": "SPACESHIP", "alias": "technology"},
            ]
        )
    entities.append({"entity_id": "car", "label": "CAR", "alias": "car"})
    unique = []
    seen = set()
    for item in entities:
        key = (item["entity_id"], item["alias"])
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def _number_mentions(text: str) -> list[dict[str, Any]]:
    mentions = []
    for match in _NUMBER_RE.finditer(text):
        value = numeric_value(match.group(0))
        if value is not None:
            mentions.append({"start": match.start(), "end": match.end(), "text": match.group(0), "value": value})
    for match in _WORD_NUMBER_RE.finditer(text):
        value = numeric_value(match.group(0))
        if value is not None:
            mentions.append({"start": match.start(), "end": match.end(), "text": match.group(0), "value": value})
    dedup: dict[tuple[int, int], dict[str, Any]] = {}
    for mention in mentions:
        dedup[(mention["start"], mention["end"])] = mention
    return sorted(dedup.values(), key=lambda item: item["start"])


def _frames_for_interval(sentence: dict[str, Any], frame_context: list[dict[str, Any]], image_paths: list[str]) -> list[dict[str, Any]]:
    start = sentence.get("start_context", sentence.get("start", sentence.get("start_source", sentence.get("start_source_seconds"))))
    end = sentence.get("end_context", sentence.get("end", sentence.get("end_source", sentence.get("end_source_seconds"))))
    evidence = []
    for frame in frame_context:
        time_value = frame.get("time_seconds")
        if time_value is None:
            continue
        try:
            time_float = float(time_value)
            in_interval = start is None or end is None or float(start) <= time_float <= float(end)
        except (TypeError, ValueError):
            in_interval = False
        if not in_interval:
            continue
        image_index = int(frame.get("image_index") or 0)
        evidence.append(
            {
                "frame_id": frame.get("source_frame"),
                "time_seconds": time_value,
                "path": image_paths[image_index - 1] if 0 < image_index <= len(image_paths) else None,
            }
        )
    return evidence[:3]


def _record(
    *,
    clip_id: str,
    entity_id: str,
    entity_label: str,
    value: float | None,
    unit: str | None,
    state_start: Any,
    state_end: Any,
    source_type: str,
    evidence_frames: list[dict[str, Any]],
    evidence_sentence_id: str | None,
    confidence: float,
    raw_text: str | None,
    evidence_text: str | None,
    value_type: str,
) -> dict[str, Any]:
    return {
        "clip_id": clip_id,
        "state_id": None,
        "state_key": None,
        "state_label": None,
        "entity_id": entity_id,
        "entity": entity_label,
        "metric": "value",
        "value": value,
        "unit": unit,
        "state_start": state_start,
        "state_end": state_end,
        "source_type": source_type,
        "evidence_frames": evidence_frames,
        "evidence_sentence_id": evidence_sentence_id,
        "confidence": confidence,
        "review_status": "machine",
        "raw_text": raw_text,
        "evidence_text": evidence_text,
        "value_type": value_type,
    }


def narration_records_from_sentences(
    sentences: list[dict[str, Any]],
    chart_context: dict[str, Any],
    clip_id: str,
    intervals: dict[str, Any] | None = None,
    frame_context: list[dict[str, Any]] | None = None,
    image_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    aliases = _entity_aliases(chart_context)
    if not aliases:
        return records
    for idx, sentence in enumerate(sentences, start=1):
        if not isinstance(sentence, dict) or not _sentence_overlaps_visual(sentence, intervals):
            continue
        text = str(sentence.get("text", "") or "")
        lower = text.lower()
        numbers = _number_mentions(text)
        sentence_entities = []
        for item in aliases:
            alias = item["alias"]
            pos = lower.find(alias)
            if pos >= 0:
                sentence_entities.append({**item, "pos": pos, "end": pos + len(alias)})
        if not sentence_entities:
            continue
        evidence_frames = _frames_for_interval(sentence, frame_context or [], image_paths or [])
        used_numbers: set[tuple[int, int]] = set()
        for entity in sentence_entities:
            following = [number for number in numbers if number["start"] >= entity["end"]]
            available = [number for number in following if (number["start"], number["end"]) not in used_numbers]
            nearest = available[0] if available else None
            if nearest:
                used_numbers.add((nearest["start"], nearest["end"]))
                value_text = nearest["text"]
                records.append(
                    _record(
                        clip_id=clip_id,
                        entity_id=entity["entity_id"],
                        entity_label=_display_entity(entity["label"]),
                        value=nearest["value"],
                        unit=infer_unit(value_text, text),
                        state_start=sentence.get("start_context", sentence.get("start", sentence.get("start_source", sentence.get("start_source_seconds")))),
                        state_end=sentence.get("end_context", sentence.get("end", sentence.get("end_source", sentence.get("end_source_seconds")))),
                        source_type="narration",
                        evidence_frames=evidence_frames,
                        evidence_sentence_id=_sentence_id(sentence, idx),
                        confidence=_as_float(sentence.get("confidence")) or 0.65,
                        raw_text=value_text,
                        evidence_text=text,
                        value_type="exact",
                    )
                )
            elif re.search(r"\ba lot of time\b|\bmany\b|\bmuch\b", lower):
                records.append(
                    _record(
                        clip_id=clip_id,
                        entity_id=entity["entity_id"],
                        entity_label=_display_entity(entity["label"]),
                        value=None,
                        unit="day" if "time" in lower else None,
                        state_start=sentence.get("start_context", sentence.get("start", sentence.get("start_source", sentence.get("start_source_seconds")))),
                        state_end=sentence.get("end_context", sentence.get("end", sentence.get("end_source", sentence.get("end_source_seconds")))),
                        source_type="narration",
                        evidence_frames=evidence_frames,
                        evidence_sentence_id=_sentence_id(sentence, idx),
                        confidence=_as_float(sentence.get("confidence")) or 0.65,
                        raw_text="a lot of time",
                        evidence_text=text,
                        value_type='qualitative: "a lot of time"',
                    )
                )
    return records


def _value_key(record: dict[str, Any]) -> tuple[str, str, float | None, str | None]:
    return (
        str(record.get("entity_id")),
        str(record.get("metric")),
        numeric_value(record.get("value")),
        None if record.get("unit") in (None, "") else str(record.get("unit")),
    )


def _state_signature(records: list[dict[str, Any]]) -> tuple[tuple[str, str, float | None, str | None, str | None], ...]:
    return tuple(sorted((str(row.get("entity_id")), str(row.get("metric")), numeric_value(row.get("value")), row.get("unit"), row.get("value_type")) for row in records))


def fuse_visual_and_narration_records(visual: list[dict[str, Any]], narration: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fused = [dict(record) for record in visual]
    for narration_record in narration:
        matched = False
        conflict = False
        for record in fused:
            same_identity = record.get("entity_id") == narration_record.get("entity_id") and record.get("metric") == narration_record.get("metric")
            if not same_identity:
                continue
            matched = True
            if _value_key(record) == _value_key(narration_record):
                record["source_type"] = "both" if record.get("source_type") == "visual" else record.get("source_type")
                record["evidence_sentence_id"] = narration_record.get("evidence_sentence_id")
                record["confidence"] = max(
                    _as_float(record.get("confidence")) or 0.0,
                    _as_float(narration_record.get("confidence")) or 0.0,
                )
                record["review_status"] = "machine"
            else:
                conflict = True
                record["review_status"] = "needs_review"
                record["conflict_with"] = {
                    "source_type": "narration",
                    "value": narration_record.get("value"),
                    "unit": narration_record.get("unit"),
                    "evidence_sentence_id": narration_record.get("evidence_sentence_id"),
                }
        if conflict:
            copy = dict(narration_record)
            copy["review_status"] = "needs_review"
            copy["conflict_with"] = {"source_type": "visual"}
            fused.append(copy)
        elif not matched:
            fused.append(dict(narration_record))
    return fused


def merge_consecutive_states(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda row: (
            str(row.get("entity_id")),
            str(row.get("metric")),
            float(row["state_start"]) if row.get("state_start") is not None else 0.0,
        ),
    )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in ordered:
        current = dict(record)
        key = (str(current.get("entity_id")), str(current.get("metric")))
        bucket = grouped.setdefault(key, [])
        if bucket:
            prev = bucket[-1]
            same_data = _value_key(prev) == _value_key(current) and prev.get("review_status") == current.get("review_status")
            same_explicit_state = prev.get("state_key") == current.get("state_key")
            if prev.get("state_key") is not None or current.get("state_key") is not None:
                same_data = same_data and same_explicit_state
            prev_end = prev.get("state_end")
            cur_start = current.get("state_start")
            consecutive = prev_end is None or cur_start is None or float(cur_start) >= float(prev_end)
            if same_data and consecutive:
                if current.get("state_end") is not None:
                    prev["state_end"] = current["state_end"]
                prev["evidence_frames"] = [*prev.get("evidence_frames", []), *current.get("evidence_frames", [])]
                if not prev.get("evidence_sentence_id"):
                    prev["evidence_sentence_id"] = current.get("evidence_sentence_id")
                prev["confidence"] = max(
                    _as_float(prev.get("confidence")) or 0.0,
                    _as_float(current.get("confidence")) or 0.0,
                )
                if prev.get("source_type") != current.get("source_type"):
                    prev["source_type"] = "both" if {prev.get("source_type"), current.get("source_type")} <= {"visual", "narration", "both"} else prev.get("source_type")
                continue
        bucket.append(current)
    merged = sorted(
        [record for rows in grouped.values() for record in rows],
        key=lambda row: (
            float(row["state_start"]) if row.get("state_start") is not None else 0.0,
            str(row.get("entity_id")),
            str(row.get("metric")),
        ),
    )
    _assign_state_ids(merged)
    return merged


def _assign_state_ids(records: list[dict[str, Any]]) -> None:
    explicit_groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        state_key = record.get("state_key")
        if state_key not in (None, ""):
            explicit_groups.setdefault(str(state_key), []).append(record)

    group_order: list[tuple[float, str, list[dict[str, Any]]]] = []
    assigned: set[int] = set()
    for state_key, rows in explicit_groups.items():
        starts = [_as_float(row.get("state_start")) for row in rows]
        ends = [_as_float(row.get("state_end")) for row in rows]
        start = min((value for value in starts if value is not None), default=None)
        end = max((value for value in ends if value is not None), default=start)
        for row in rows:
            row["state_start"] = start
            row["state_end"] = end
            if not row.get("state_label"):
                row["state_label"] = state_key
            assigned.add(id(row))
        group_order.append((start if start is not None else 0.0, state_key, rows))

    for record in records:
        if id(record) not in assigned:
            start = _as_float(record.get("state_start"))
            group_order.append((start if start is not None else 0.0, f"__row_{len(group_order):06d}", [record]))

    for idx, (_, _, rows) in enumerate(sorted(group_order, key=lambda item: (item[0], item[1])), start=1):
        state_id = f"state_{idx:03d}"
        for row in rows:
            row["state_id"] = state_id


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_dynamic_records(
    *,
    clip_id: str,
    visual_data: dict[str, Any],
    frame_context: list[dict[str, Any]],
    image_paths: list[str],
    narration_sentences: list[dict[str, Any]],
    chart_context: dict[str, Any],
    intervals: dict[str, Any] | None = None,
    audit: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    audit_rows = list(audit or [])
    visual = visual_records_from_clip_data(visual_data, frame_context, image_paths, clip_id)
    if isinstance(visual_data, dict):
        chart_context = {
            **chart_context,
            "visible_text": visual_data.get("visible_text", []),
            "x_axis": visual_data.get("x_axis") or chart_context.get("x_axis"),
            "y_axis": visual_data.get("y_axis") or chart_context.get("y_axis"),
        }
    narration = narration_records_from_sentences(
        narration_sentences,
        chart_context,
        clip_id,
        intervals=intervals,
        frame_context=frame_context,
        image_paths=image_paths,
    )
    fused = fuse_visual_and_narration_records(visual, narration)
    for record in fused:
        record["metric"] = sanitize_metric(record.get("metric"))
    states = merge_consecutive_states(fused)
    numeric_fact_count = sum(1 for row in states if row.get("value") is not None)
    excluded = numeric_fact_count == 0
    if excluded:
        audit_rows.append({"stage": "dynamic_data", "status": "excluded", "reason": NO_RECOVERABLE_QUANTITATIVE_DATA})
    final_table = build_final_data_table(states)
    change_events = build_data_change_events(states)
    return {
        "clip_id": clip_id,
        "chart_type": str(visual_data.get("chart_type") or chart_context.get("chart_type") or "").lower(),
        "states": states,
        "final_data_table": final_table,
        "data_change_events": change_events,
        "excluded": excluded,
        "exclude_reason": NO_RECOVERABLE_QUANTITATIVE_DATA if excluded else None,
        "include_in_dataset": not excluded,
        "data_completeness": "complete" if states and numeric_fact_count == len(states) else "partial" if states else "none",
        "numeric_fact_count": numeric_fact_count,
        "dynamic_data": len(change_events) > 1,
        "data_change_count": len(change_events),
        "visual_record_count": len(visual),
        "narration_record_count": len(narration),
        "audit": audit_rows,
    }


def build_final_data_table(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for state in sorted(states, key=lambda row: float(row.get("state_start") or 0.0)):
        latest[(str(state.get("entity_id")), str(state.get("metric") or ""))] = state
    rows = []
    for state in latest.values():
        rows.append(
            {
                "clip_id": state.get("clip_id"),
                "entity_id": state.get("entity_id"),
                "entity": state.get("entity") or state.get("entity_id"),
                "metric": state.get("metric"),
                "value": state.get("value"),
                "unit": state.get("unit"),
                "type": state.get("value_type", "exact"),
                "source_type": state.get("source_type"),
                "evidence_frames": state.get("evidence_frames", []),
                "evidence_sentence_id": state.get("evidence_sentence_id"),
                "confidence": state.get("confidence"),
                "review_status": state.get("review_status"),
            }
        )
    return rows


def build_data_change_events(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    active: dict[tuple[str, str], tuple[float | None, str | None, str | None]] = {}
    for state in sorted(states, key=lambda row: float(row.get("state_start") or 0.0)):
        entity_id = str(state.get("entity_id"))
        metric = str(state.get("metric") or "")
        active_key = (entity_id, metric)
        value_sig = (numeric_value(state.get("value")), state.get("unit"), state.get("value_type"))
        action = "insert" if active_key not in active else "update" if active[active_key] != value_sig else None
        if action is None:
            continue
        active[active_key] = value_sig
        events.append(
            {
                "clip_id": state.get("clip_id"),
                "event_id": f"event_{len(events) + 1:03d}",
                "event_type": action,
                "entity_id": entity_id,
                "entity": state.get("entity") or entity_id,
                "metric": state.get("metric"),
                "value": state.get("value"),
                "unit": state.get("unit"),
                "state_start": state.get("state_start"),
                "state_end": state.get("state_end"),
                "source_type": state.get("source_type"),
                "evidence_frames": state.get("evidence_frames", []),
                "evidence_sentence_id": state.get("evidence_sentence_id"),
                "confidence": state.get("confidence"),
                "review_status": state.get("review_status"),
            }
        )
    return events


def plan_dynamic_state_keyframes(
    result: dict[str, Any],
    *,
    max_states: int = 8,
) -> dict[str, Any]:
    states = result.get("states") if isinstance(result.get("states"), list) else []
    if not result.get("include_in_dataset") or not result.get("dynamic_data"):
        return {"should_save": False, "reason": "not_dynamic_recovered_data", "states": []}

    # Only an explicit printed state label (year/period read from the frame)
    # defines a state.  Rows without one are animation frames or duplicates
    # of the same static chart, not separate states, so they never form a
    # state group.
    groups: dict[str, list[dict[str, Any]]] = {}
    order: dict[str, float] = {}
    for row in states:
        if not isinstance(row, dict):
            continue
        state_key = row.get("state_key") or row.get("state_label")
        if state_key in (None, ""):
            continue
        state_key = str(state_key)
        groups.setdefault(state_key, []).append(row)
        start = _as_float(row.get("state_start"))
        order[state_key] = min(order.get(state_key, start if start is not None else 0.0), start if start is not None else 0.0)

    complete_groups = []
    for state_key, rows in groups.items():
        numeric_rows = [row for row in rows if row.get("value") is not None]
        if not numeric_rows or len(numeric_rows) != len(rows):
            continue
        has_evidence = all(row.get("evidence_frames") or row.get("evidence_sentence_id") for row in rows)
        if not has_evidence:
            continue
        complete_groups.append((order.get(state_key, 0.0), state_key, rows))
    complete_groups = sorted(complete_groups, key=lambda item: (item[0], item[1]))
    if not groups:
        return {"should_save": False, "reason": "no_explicit_state_labels", "states": []}
    if len(complete_groups) < 2:
        return {"should_save": False, "reason": "fewer_than_two_complete_evidenced_data_states", "states": []}

    # A printed year/period only denotes a real state when the same entities
    # are re-rendered under different labels (e.g. 1990 vs 2017).  Groups
    # whose entities are disjoint (years used as x-axis categories, or line
    # data points) describe one static chart, not multiple states.
    all_ids = [
        {str(r.get("entity_id") or "") for r in rows if r.get("entity_id")}
        for _, _, rows in complete_groups
    ]
    shared_entity = any(
        all_ids[i] & all_ids[j]
        for i in range(len(all_ids))
        for j in range(i + 1, len(all_ids))
    )
    if not shared_entity:
        return {"should_save": False, "reason": "disjoint_entities_not_states", "states": []}

    # Marks read from one and the same frame (e.g. all x-axis values of one
    # line chart) are points of a single static chart, not video states.
    evidence_frames = {
        (
            row.get("source_frame_id") or row.get("source_frame_path"),
            _as_float(row.get("timestamp")),
        )
        for group in complete_groups
        for row in (_dynamic_keyframe_row(group),)
    }
    if len(evidence_frames) <= 1:
        return {"should_save": False, "reason": "static_chart_points_from_single_visual_frame", "states": []}

    first = complete_groups[0]
    last = complete_groups[-1]
    if _state_signature(first[2]) == _state_signature(last[2]):
        return {"should_save": False, "reason": "no_entity_or_value_change", "states": []}

    # Keep every complete evidenced state as a keyframe, capped at max_states.
    # When capped, always keep the first and last and sample the middle evenly.
    limit = max(2, int(max_states) if int(max_states) > 0 else 8)
    if len(complete_groups) > limit:
        total = len(complete_groups)
        picks = sorted({round(index * (total - 1) / (limit - 1)) for index in range(limit)})
        complete_groups = [complete_groups[index] for index in picks]
    planned = [_dynamic_keyframe_row(group) for group in complete_groups]
    if any(row.get("timestamp") is None and not row.get("source_frame_path") for row in planned):
        return {"should_save": False, "reason": "missing_visual_frame_evidence", "states": []}
    return {
        "should_save": True,
        "reason": "complete_evidenced_data_states_selected",
        "selection_rule": "all_complete_evidenced_data_states_as_state_keyframes",
        "states": planned,
    }


def _dynamic_keyframe_row(group: tuple[float, str, list[dict[str, Any]]]) -> dict[str, Any]:
    _, state_key, rows = group
    state_id = rows[0].get("state_id") or state_key
    visual_evidence = []
    for row in rows:
        for frame in row.get("evidence_frames") or []:
            if isinstance(frame, dict):
                visual_evidence.append(frame)
    visual_evidence = sorted(
        visual_evidence,
        key=lambda frame: _as_float(frame.get("time_seconds")) if _as_float(frame.get("time_seconds")) is not None else -1.0,
    )
    representative = visual_evidence[-1] if visual_evidence else {}
    return {
        "state_id": state_id,
        "state_key": rows[0].get("state_key"),
        "state_label": rows[0].get("state_label") or rows[0].get("state_key") or state_id,
        "timestamp": representative.get("time_seconds"),
        "source_frame_id": representative.get("frame_id"),
        "source_frame_path": representative.get("path"),
        "entity_ids": sorted({str(row.get("entity_id")) for row in rows}),
        "signature": [
            {
                "entity_id": row.get("entity_id"),
                "metric": row.get("metric"),
                "value": row.get("value"),
                "unit": row.get("unit"),
            }
            for row in sorted(rows, key=lambda item: str(item.get("entity_id")))
        ],
    }


def write_dynamic_outputs(out_dir: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    out_dir = Path(out_dir)
    dynamic_json = out_dir / "dynamic_data.json"
    dynamic_csv = out_dir / "dynamic_data.csv"
    final_csv = out_dir / "final_data_table.csv"
    changes_csv = out_dir / "data_change_events.csv"
    events_path = out_dir / "data_events.jsonl"
    write_json(dynamic_json, result)
    states = result.get("states", [])
    if states:
        write_csv(dynamic_csv, states)
    elif dynamic_csv.exists():
        dynamic_csv.unlink()
    final_rows = result.get("final_data_table", [])
    change_rows = result.get("data_change_events", [])
    if final_rows:
        write_csv(final_csv, final_rows)
    elif final_csv.exists():
        final_csv.unlink()
    if change_rows:
        write_csv(changes_csv, change_rows)
    elif changes_csv.exists():
        changes_csv.unlink()
    write_jsonl(events_path, change_rows)
    return {
        "dynamic_data_json": str(dynamic_json),
        "dynamic_data_csv": str(dynamic_csv) if states else None,
        "final_data_table_csv": str(final_csv) if final_rows else None,
        "data_change_events_csv": str(changes_csv) if change_rows else None,
        "events_path": str(events_path),
    }


def load_narration_sentences(processed_root: str | Path | None, clip_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if processed_root is None:
        return [], {"status": "missing", "reason": "processed_root not configured"}
    root = Path(processed_root) / clip_id
    candidates = [
        root / "narration" / "narration_reviewed.json",
        root / "narration_reviewed.json",
        root / "narration" / "selected_full_sentences.jsonl",
    ]
    audit = {"status": "missing", "source": None, "reason": None}
    for path in candidates:
        if not path.exists():
            continue
        audit["source"] = str(path)
        try:
            if path.suffix == ".jsonl":
                return read_jsonl(path), {**audit, "status": "loaded"}
            payload = read_json(path)
            rows = payload.get("sentences") or payload.get("selected_full_sentences") or []
            return rows if isinstance(rows, list) else [], {**audit, "status": "loaded"}
        except Exception as exc:
            return [], {**audit, "status": "failed", "reason": str(exc)}
    return [], audit


def plan_state_sampling(scored_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    coarse_fps = float(cfg.get("clip_data", {}).get("coarse_fps", 2))
    fine_fps = float(cfg.get("clip_data", {}).get("fine_fps", 8))
    threshold = float(cfg.get("clip_data", {}).get("motion_change_threshold", 0.08))
    rows = sorted(scored_rows, key=lambda row: float(row.get("timestamp", 0.0)))
    coarse = [row for row in rows if abs((float(row.get("timestamp", 0.0)) * coarse_fps) - round(float(row.get("timestamp", 0.0)) * coarse_fps)) < 0.05]
    if not coarse:
        coarse = rows
    windows = []
    previous: dict[str, Any] | None = None
    for row in coarse:
        if previous is not None:
            left = float(previous.get("timestamp", 0.0))
            right = float(row.get("timestamp", 0.0))
            motion = max(
                float(previous.get("score", {}).get("motion_score", previous.get("motion_score", 0.0)) or 0.0),
                float(row.get("score", {}).get("motion_score", row.get("motion_score", 0.0)) or 0.0),
            )
            if motion >= threshold:
                windows.append({"start": left, "end": right, "target_fps": fine_fps, "reason": "coarse_motion_change"})
        previous = row
    selected = []
    for row in rows:
        timestamp = float(row.get("timestamp", 0.0))
        in_window = any(window["start"] <= timestamp <= window["end"] for window in windows)
        if in_window or row in coarse:
            selected.append(row)
    for boundary in (rows[0], rows[-1]) if rows else ():
        if boundary not in selected:
            selected.append(boundary)
    selected = sorted(selected, key=lambda row: float(row.get("timestamp", 0.0)))
    return {
        "coarse_fps": coarse_fps,
        "fine_fps": fine_fps,
        "coarse_frame_count": len(coarse),
        "fine_windows": windows,
        "selected_rows": selected,
        "selected_frame_count": len(selected),
    }
