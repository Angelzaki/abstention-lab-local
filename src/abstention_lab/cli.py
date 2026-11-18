from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from abstention_lab.calibrate import calibrate

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="abstention-lab")
    parser.add_argument("--docs", default="samples/docs")
    parser.add_argument("--cases", default="samples/cases.jsonl")
    parser.add_argument("--results", default="results")
    args = parser.parse_args(argv)

    report = calibrate(Path(args.docs), Path(args.cases))
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "abstention_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    table = Table(title="Abstention Lab · top 5")
    table.add_column("min_score", justify="right")
    table.add_column("min_overlap", justify="right")
    table.add_column("accuracy", justify="right")
    for row in sorted(report["grid"], key=lambda x: x["accuracy"], reverse=True)[:5]:
        table.add_row(str(row["min_score"]), str(row["min_overlap"]), f"{row['accuracy']:.0%}")
    console.print(table)
    best = report["best"]
    console.print(
        f"best: score=[bold]{best['min_score']}[/bold] overlap=[bold]{best['min_overlap']}[/bold] "
        f"({best['accuracy']:.0%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
