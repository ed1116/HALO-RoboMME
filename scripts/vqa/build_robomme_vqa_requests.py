#!/usr/bin/env python3
"""Build a deterministic, task-balanced RoboMME VQA request pilot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from halo.robomme_vqa.request_builder import build_pilot_requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--shared-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-requests", type=int, default=96)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-evidence-frames", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_pilot_requests(
        args.raw_root,
        args.shared_manifest,
        args.output_dir,
        target_requests=args.target_requests,
        split_seed=args.split_seed,
        seed=args.seed,
        max_evidence_frames=args.max_evidence_frames,
    )
    print(output)


if __name__ == "__main__":
    main()
