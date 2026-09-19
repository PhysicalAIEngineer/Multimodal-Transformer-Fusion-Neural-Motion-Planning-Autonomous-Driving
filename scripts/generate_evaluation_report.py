from __future__ import annotations

"""Generate a publication/portfolio-friendly evaluation report from JSON output."""

import argparse
import json
from pathlib import Path


def markdown_table(section: dict) -> str:
    lines = ["| Metric | Measured value |", "| --- | ---: |"]
    for key, value in section.items():
        if value is not None:
            lines.append(f"| {key} | {value} |")
    return "
".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    lines = [
        "# CARLA Evaluation Report",
        "",
        "Generated from measured evaluation records. Missing metrics are not inferred.",
        "",
    ]
    for section_name, section in report.items():
        if isinstance(section, dict):
            lines += [f"## {section_name.title()}", "", markdown_table(section), ""]
    Path(args.output).write_text("
".join(lines), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
