from __future__ import annotations

import argparse
import json
from pathlib import Path

from production.evaluation import aggregate_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate CARLA evaluation records from JSONL."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to JSONL evaluation records.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the aggregated JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = aggregate_jsonl(args.input, args.output)
    print(json.dumps(report, indent=2))
    print(f"Saved report to {Path(args.output)}")


if __name__ == "__main__":
    main()
