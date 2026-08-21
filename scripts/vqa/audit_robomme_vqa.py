#!/usr/bin/env python3
"""Create a deterministic human-audit packet without editing corpus records."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from halo.robomme_vqa.audit import attach_visual_evidence, read_request_artifact
from halo.robomme_vqa.contracts import parse_record, strict_json_loads
from halo.robomme_vqa.io import guarded_output_directory, write_json, write_jsonl
from halo.robomme_vqa.prompts import TASK_INSTRUCTIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("pilot", "pretrain"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-stratum", type=int, default=1)
    return parser.parse_args()


def read_records(payload: bytes) -> list[dict[str, Any]]:
    """Parse the exact bytes that were hashed, so provenance cannot drift."""
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(parse_record(strict_json_loads(line)))
        except ValueError as error:
            raise ValueError(f"invalid VQA record on line {line_number}: {error}") from error
    if not records:
        raise ValueError("records JSONL is empty")
    return records


def sample_group(
    rows: list[dict[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row["task_name"], row["episode_id"], row["query_timestep"], row["candidate_id"]
        ),
    )
    if len(ordered) <= count:
        return ordered
    return rng.sample(ordered, count)


def main() -> None:
    args = parse_args()
    if args.per_stratum <= 0:
        raise ValueError("--per-stratum must be positive")
    output_dir = guarded_output_directory(args.output_dir)
    records_bytes = args.records_jsonl.read_bytes()
    records = read_records(records_bytes)
    request_artifact = read_request_artifact(
        args.requests_jsonl, args.request_manifest
    )
    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    deficits: list[dict[str, Any]] = []

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    if args.stage == "pilot":
        for row in records:
            groups[(row["suite_name"], bool(row["accepted"]))].append(row)
        for suite_name in ("counting", "permanence", "reference", "imitation"):
            for accepted in (False, True):
                key = (suite_name, accepted)
                rows = groups[key]
                requested = 20 if accepted else 10
                selected.extend(sample_group(rows, requested, rng))
                if len(rows) < requested:
                    deficits.append(
                        {
                            "stratum": list(key),
                            "requested": requested,
                            "available": len(rows),
                        }
                    )
    else:
        for row in records:
            groups[(row["task_name"], row["question_family"])].append(row)
        for task_name, (_, families) in TASK_INSTRUCTIONS.items():
            for family in families:
                key = (task_name, family)
                rows = groups[key]
                selected.extend(sample_group(rows, args.per_stratum, rng))
                if len(rows) < args.per_stratum:
                    deficits.append(
                        {
                            "stratum": list(key),
                            "requested": args.per_stratum,
                            "available": len(rows),
                        }
                    )

    packet = []
    for row in selected:
        audited = attach_visual_evidence(row, request_artifact)
        packet.append(
            {
                **audited,
                "human_audit": {
                    "status": "pending",
                    "error_classes": [],
                    "notes": "",
                },
            }
        )
    # Complete every source join before creating an output directory.
    output_dir.mkdir(parents=True, exist_ok=False)
    write_jsonl(output_dir / "audit_packet.jsonl", packet)
    write_json(
        output_dir / "audit_manifest.json",
        {
            "schema_version": "halo.robomme.vqa.audit-manifest/v1",
            "stage": args.stage,
            "seed": args.seed,
            "source_records": str(args.records_jsonl.resolve()),
            "source_records_sha256": hashlib.sha256(records_bytes).hexdigest(),
            "source_record_count": len(records),
            "source_requests": str(args.requests_jsonl.resolve()),
            "source_requests_sha256": request_artifact.requests_sha256,
            "source_request_count": len(request_artifact.requests),
            "source_request_manifest": str(args.request_manifest.resolve()),
            "source_request_manifest_sha256": request_artifact.manifest_sha256,
            "sample_count": len(packet),
            "deficits": deficits,
            "review_status": "pending_human_review",
            "editing_rule": (
                "Do not overwrite source records or images. Image fields are read-only "
                "references; record error classes and notes only in this packet."
            ),
        },
    )


if __name__ == "__main__":
    main()
