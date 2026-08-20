#!/usr/bin/env python3
"""Run a bounded RoboMME VQA pilot into a new /data artifact directory."""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from halo.robomme_vqa.contracts import load_model_config
from halo.robomme_vqa.io import (
    guarded_output_directory,
    read_request_jsonl,
    write_json,
    write_jsonl,
)
from halo.robomme_vqa.pipeline import CandidateGenerator, IndependentJudge, OfflineVQAPipeline
from halo.robomme_vqa.qwen import Qwen3VLBackend


DEFAULT_CONFIG = REPOSITORY_ROOT / "config/vqa/robomme_qwen3_vl_8b_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--candidate-count", type=int, default=3)
    parser.add_argument("--max-evidence-frames", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face downloads into the existing cache; default is cache-only.",
    )
    return parser.parse_args()


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
    output_dir.mkdir(parents=True, exist_ok=False)

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

    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    requests_processed = 0
    start_time = time.perf_counter()
    for request_index, request in enumerate(read_request_jsonl(args.requests_jsonl)):
        if request_index == args.max_requests:
            break
        requests_processed += 1
        try:
            records.extend(pipeline.process(request, candidate_count=args.candidate_count))
        except Exception as error:
            failures.append(
                {
                    "request_index": request_index,
                    "source_request_sha256": request.source_sha256(),
                    "error_type": type(error).__name__,
                    "error": str(error)[:512],
                }
            )

    elapsed_seconds = time.perf_counter() - start_time
    records_path = output_dir / "records.jsonl"
    write_jsonl(records_path, records)
    accepted = sum(bool(record["accepted"]) for record in records)
    manifest = {
        "schema_version": "halo.robomme.vqa.pilot-manifest/v1",
        "requests_jsonl": str(args.requests_jsonl.resolve()),
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
        "elapsed_seconds": elapsed_seconds,
        "requests_per_second": (
            requests_processed / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "peak_vram_bytes": (
            torch.cuda.max_memory_allocated(args.device) if uses_cuda else None
        ),
        "records_size_bytes": records_path.stat().st_size,
        "camera_identity_review": "pending_human_audit",
        "allow_download": args.allow_download,
        "device": args.device,
    }
    write_json(output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
