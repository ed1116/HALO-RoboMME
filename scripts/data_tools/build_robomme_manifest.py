"""Build the reusable metadata manifest for raw RoboMME HDF5 files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from halo.robomme import write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Directory with 16 HDF5 files"
    )
    parser.add_argument("--output", type=Path, required=True, help="New manifest JSON path")
    parser.add_argument(
        "--validate-demo-prefix",
        action="store_true",
        help="Read every demo flag to verify one contiguous prefix (substantially slower)",
    )
    args = parser.parse_args()
    output = write_manifest(
        args.root,
        args.output,
        validate_demo_prefix=args.validate_demo_prefix,
    )
    print(output)


if __name__ == "__main__":
    main()
