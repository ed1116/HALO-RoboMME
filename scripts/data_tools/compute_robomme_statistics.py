#!/usr/bin/env python3
"""Compute training-only RoboMME state/action normalization statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from halo.robomme import RoboMMEHDF5Dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--dev-episodes-per-task", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing statistics: {args.output}")

    dataset = RoboMMEHDF5Dataset(
        args.root,
        horizon=args.horizon,
        split="train",
        split_seed=args.split_seed,
        dev_episodes_per_task=args.dev_episodes_per_task,
        manifest_path=args.manifest,
    )
    statistics = dataset.compute_training_statistics()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(statistics.to_json() + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
