from __future__ import annotations

"""Plot measured evaluation results without inventing missing values."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def flatten(prefix: str, value: dict[str, object], output: dict[str, float]) -> None:
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flatten(name, item, output)
        elif isinstance(item, (int, float)):
            output[name] = float(item)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a measured evaluation JSON report.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    values: dict[str, float] = {}
    flatten("", report, values)
    if not values:
        raise SystemExit("No numeric measured metrics were found.")

    labels = list(values)
    numbers = [values[label] for label in labels]
    height = max(4.0, 0.32 * len(labels))

    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(labels, numbers)
    ax.set_title("CARLA Evaluation Metrics")
    ax.set_xlabel("Measured value")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "evaluation_metrics.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
