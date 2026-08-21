"""Strict request parsing and guarded artifact output paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .contracts import ContractError, strict_json_loads
from .pipeline import VQARequest
from .timeline import TimelineFrame


REQUEST_SCHEMA_VERSION = "halo.robomme.vqa.request/v1"
VQA_OUTPUT_ROOT = Path("/data/ed1116/robomme/vqa/halo")


def guarded_output_directory(path: str | Path) -> Path:
    """Require a new, explicit directory below the approved artifact root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        raise ValueError("output directory must be an explicit absolute path")
    resolved = candidate.resolve(strict=False)
    root = VQA_OUTPUT_ROOT.resolve(strict=False)
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"output directory must be below {root}")
    return resolved


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ContractError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def request_from_dict(payload: Mapping[str, Any]) -> VQARequest:
    expected = {
        "schema_version",
        "task_name",
        "suite_name",
        "episode_id",
        "task_goal",
        "query_timestep",
        "timeline",
        "known_counts_by_family",
    }
    _exact_keys(payload, expected, "request")
    if payload["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise ContractError("unsupported request schema_version")
    raw_timeline = payload["timeline"]
    if not isinstance(raw_timeline, list) or not raw_timeline:
        raise ContractError("timeline must be a non-empty list")
    timeline: list[TimelineFrame] = []
    frame_fields = {
        "timestep",
        "front_image",
        "wrist_image",
        "event_boundary",
        "change_score",
    }
    for index, row in enumerate(raw_timeline):
        if not isinstance(row, dict):
            raise ContractError(f"timeline frame {index} must be an object")
        _exact_keys(row, frame_fields, f"timeline frame {index}")
        if not isinstance(row["event_boundary"], bool):
            raise ContractError("event_boundary must be boolean")
        if isinstance(row["change_score"], bool) or not isinstance(
            row["change_score"], (int, float)
        ):
            raise ContractError("change_score must be numeric")
        try:
            timeline.append(
                TimelineFrame(
                    timestep=row["timestep"],
                    front_image=row["front_image"],
                    wrist_image=row["wrist_image"],
                    event_boundary=row["event_boundary"],
                    change_score=float(row["change_score"]),
                )
            )
        except (TypeError, ValueError) as error:
            raise ContractError(f"invalid timeline frame {index}: {error}") from error
    known_counts = payload["known_counts_by_family"]
    if not isinstance(known_counts, dict):
        raise ContractError("known_counts_by_family must be an object")
    try:
        return VQARequest(
            task_name=payload["task_name"],
            suite_name=payload["suite_name"],
            episode_id=payload["episode_id"],
            task_goal=payload["task_goal"],
            query_timestep=payload["query_timestep"],
            timeline=tuple(timeline),
            known_counts_by_family=dict(known_counts),
        )
    except (TypeError, ValueError) as error:
        raise ContractError(f"invalid request: {error}") from error


def parse_request_jsonl(text: str) -> Iterable[VQARequest]:
    """Parse requests from already-read text, so callers can hash the same bytes."""
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = strict_json_loads(line)
        except ContractError as error:
            raise ContractError(f"invalid JSON on line {line_number}: {error}") from error
        if not isinstance(payload, dict):
            raise ContractError(f"request line {line_number} must be an object")
        yield request_from_dict(payload)


def read_request_jsonl(path: str | Path) -> Iterable[VQARequest]:
    return parse_request_jsonl(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
