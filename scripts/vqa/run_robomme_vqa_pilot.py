#!/usr/bin/env python3
"""Run a bounded, resumable RoboMME VQA pilot into a /data artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from halo.robomme_vqa.audit import read_request_artifact
from halo.robomme_vqa.contracts import ContractError, load_model_config, strict_json_loads
from halo.robomme_vqa.io import guarded_output_directory, write_json
from halo.robomme_vqa.pipeline import CandidateGenerator, IndependentJudge, OfflineVQAPipeline
from halo.robomme_vqa.qwen import Qwen3VLBackend


DEFAULT_CONFIG = REPOSITORY_ROOT / "config/vqa/robomme_qwen3_vl_8b_v1.json"
PROGRESS_SCHEMA_VERSION = "halo.robomme.vqa.pilot-progress/v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--max-evidence-frames", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run in an existing, unfinished output directory.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face downloads into the existing cache; default is cache-only.",
    )
    return parser.parse_args()


def read_whole_lines(path: Path) -> list[dict[str, Any]]:
    """Read complete JSONL rows, discarding a trailing line left by a crash."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = strict_json_loads(line)
        except ContractError:
            break  # Only the final line can be partial; stop at it.
        if not isinstance(value, dict):
            break
        rows.append(value)
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append and flush to disk so completed model calls survive interruption."""
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Replace a file atomically so a crash cannot leave it half-rewritten."""
    temporary = path.with_name(f"{path.name}.rewriting")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def prepare_output_dir(output_dir: Path, *, resume: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return
    if not resume:
        raise FileExistsError(output_dir)
    if not output_dir.is_dir():
        raise NotADirectoryError(output_dir)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"{output_dir} already holds a completed pilot")


def recover_completed_work(
    records_path: Path, progress_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep only records whose request was recorded as attempted.

    Progress is appended after a request's records are already on disk, so a
    record without a progress line belongs to an interrupted request and must
    be produced again rather than double-counted.
    """
    progress = read_whole_lines(progress_path)
    records = read_whole_lines(records_path)
    attempted = {row.get("source_request_sha256") for row in progress}
    kept = [row for row in records if row.get("source_request_sha256") in attempted]
    if len(kept) != len(records):
        rewrite_jsonl(records_path, kept)
    if len(progress) != len(read_whole_lines(progress_path)):
        rewrite_jsonl(progress_path, progress)
    return kept, progress


def main() -> None:
    args = parse_args()
    if args.max_requests <= 0:
        raise ValueError("--max-requests must be positive")
    if not 1 <= args.candidate_count <= 8:
        raise ValueError("--candidate-count must be between 1 and 8")
    if args.max_evidence_frames <= 0:
        raise ValueError("--max-evidence-frames must be positive")
    output_dir = guarded_output_directory(args.output_dir)
    config_bytes = args.config.read_bytes()
    config = load_model_config(args.config)
    observed_runtime = config.verify_runtime()
    # Bind the requests to their build manifest before spending any GPU time:
    # this verifies the JSONL hash and every referenced image.
    artifact = read_request_artifact(args.requests_jsonl, args.request_manifest)
    prepare_output_dir(output_dir, resume=args.resume)

    records_path = output_dir / "records.jsonl"
    progress_path = output_dir / "progress.jsonl"
    records, progress = recover_completed_work(records_path, progress_path)
    attempted = {row.get("source_request_sha256") for row in progress}

    import torch

    uses_cuda = args.device.startswith("cuda")
    if uses_cuda:
        torch.cuda.reset_peak_memory_stats(args.device)
    # A shared loaded model conserves VRAM. Generator and judge remain distinct
    # interfaces and every judgment is a fresh call with a different prompt.
    backend = Qwen3VLBackend(
        config, device=args.device, allow_download=args.allow_download
    )
    pipeline = OfflineVQAPipeline(
        CandidateGenerator(backend, config),
        IndependentJudge(backend, config),
        max_evidence_frames=args.max_evidence_frames,
    )

    start_time = time.perf_counter()
    for request_index, (source_hash, request) in enumerate(artifact.requests.items()):
        if request_index == args.max_requests:
            break
        if source_hash in attempted:
            continue
        try:
            new_records = pipeline.process(
                request, candidate_count=args.candidate_count
            )
        except Exception as error:  # noqa: BLE001 - recorded, never silently dropped
            append_jsonl(
                progress_path,
                [
                    {
                        "schema_version": PROGRESS_SCHEMA_VERSION,
                        "request_index": request_index,
                        "source_request_sha256": source_hash,
                        "status": "error",
                        "record_count": 0,
                        "error_type": type(error).__name__,
                        "error": str(error)[:512],
                    }
                ],
            )
            progress.append({"source_request_sha256": source_hash, "status": "error"})
            attempted.add(source_hash)
            continue
        # Records reach disk before the request is marked attempted.
        append_jsonl(records_path, new_records)
        append_jsonl(
            progress_path,
            [
                {
                    "schema_version": PROGRESS_SCHEMA_VERSION,
                    "request_index": request_index,
                    "source_request_sha256": source_hash,
                    "status": "ok",
                    "record_count": len(new_records),
                }
            ],
        )
        records.extend(new_records)
        progress.append({"source_request_sha256": source_hash, "status": "ok"})
        attempted.add(source_hash)

    elapsed_seconds = time.perf_counter() - start_time
    failures = [row for row in read_whole_lines(progress_path) if row.get("status") != "ok"]
    requests_processed = len(read_whole_lines(progress_path))
    accepted = sum(bool(record["accepted"]) for record in records)
    manifest = {
        "schema_version": "halo.robomme.vqa.pilot-manifest/v1",
        "requests_jsonl": str(args.requests_jsonl.resolve()),
        "requests_jsonl_sha256": artifact.requests_sha256,
        "request_manifest": str(args.request_manifest.resolve()),
        "request_manifest_sha256": artifact.manifest_sha256,
        "request_count": len(artifact.requests),
        "requests_processed": requests_processed,
        "records": len(records),
        "accepted": accepted,
        "rejected": len(records) - accepted,
        "acceptance_rate": accepted / len(records) if records else 0.0,
        "valid_json_rate": (
            (requests_processed - len(failures)) / requests_processed
            if requests_processed
            else 0.0
        ),
        "failures": failures,
        "generator": config.model_provenance(role="generator"),
        "judge": config.model_provenance(role="judge"),
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "observed_runtime_versions": observed_runtime,
        "candidate_count": args.candidate_count,
        "max_evidence_frames": args.max_evidence_frames,
        "max_requests": args.max_requests,
        "resumed": args.resume,
        "elapsed_seconds": elapsed_seconds,
        "requests_per_second": (
            requests_processed / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(args.device) if uses_cuda else None
        ),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "records_size_bytes": records_path.stat().st_size,
        "camera_identity_review": "pending_human_audit",
        "allow_download": args.allow_download,
        "device": args.device,
    }
    write_json(output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
