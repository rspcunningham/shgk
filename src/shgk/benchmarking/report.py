from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .models import CategorySpec, load_jsonl


def _normalized(score: float, spec: CategorySpec) -> float:
    value = (score - spec.minimum) / (spec.maximum - spec.minimum)
    if not spec.higher_is_better:
        value = 1 - value
    return max(0.0, min(1.0, value))


def build_summary(paths: list[str | Path]) -> dict[str, object]:
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    specs: dict[str, CategorySpec] = {}
    category_order: list[str] = []
    used_categories: set[str] = set()
    for path in paths:
        for record in load_jsonl(path):
            key = f"{record.get('provider')}:{record.get('model')}"
            by_model[key].append(record)
            for scoring in record.get("scoring", []):
                used_categories.update(scoring.get("scores", {}))
                for raw_spec in scoring.get("category_specs", []):
                    spec = CategorySpec.model_validate(raw_spec)
                    if spec.name not in specs:
                        specs[spec.name] = spec
                        category_order.append(spec.name)

    rows: list[dict[str, object]] = []
    for model, records in by_model.items():
        category_values: dict[str, list[float]] = defaultdict(list)
        hard_failures = 0
        record_overalls: list[float] = []
        for record in records:
            merged_scores: dict[str, float] = {}
            failures: set[str] = set()
            for scoring in record.get("scoring", []):
                merged_scores.update(
                    {name: float(value) for name, value in scoring.get("scores", {}).items()}
                )
                failures.update(scoring.get("hard_failures", []))
            for name, value in merged_scores.items():
                category_values[name].append(value)
            hard_failures += len(failures)
            weighted = [
                (_normalized(merged_scores[name], specs[name]), specs[name].weight)
                for name in merged_scores
                if name in specs and specs[name].weight > 0
            ]
            total_weight = sum(weight for _, weight in weighted)
            if total_weight:
                record_overalls.append(
                    100
                    * sum(value * weight for value, weight in weighted)
                    / total_weight
                )
        category_means = {
            name: sum(category_values[name]) / len(category_values[name])
            for name in category_values
        }
        rows.append(
            {
                "model": model,
                "cases": len(records),
                "completed": sum(
                    1 for record in records if record.get("status") == "completed"
                ),
                "scoring_errors": sum(
                    1 for record in records if record.get("scoring_errors")
                ),
                "hard_failures": hard_failures,
                "overall": (
                    sum(record_overalls) / len(record_overalls)
                    if record_overalls
                    else None
                ),
                "scores": category_means,
                "coverage": {
                    name: len(values) for name, values in category_values.items()
                },
            }
        )
    rows.sort(
        key=lambda row: (
            row["overall"] is not None,
            row["overall"] if row["overall"] is not None else -1,
        ),
        reverse=True,
    )
    return {
        "categories": [
            specs[name].model_dump()
            for name in category_order
            if name in used_categories
        ],
        "models": rows,
    }


def _display_score(value: object) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def print_summary(summary: dict[str, object], *, console: Console | None = None) -> None:
    console = console or Console()
    categories = [
        CategorySpec.model_validate(value) for value in summary.get("categories", [])
    ]
    models = summary.get("models", [])
    if console.width < 140 and categories:
        table = Table(title="ChGK translation benchmark", header_style="bold cyan")
        table.add_column("Rank", justify="right", style="dim")
        table.add_column("Model", no_wrap=True)
        table.add_column("N", justify="right")
        table.add_column("Complete", justify="right")
        table.add_column("Hard fails", justify="right")
        table.add_column("Overall", justify="right", style="bold")
        for rank, row in enumerate(models, start=1):
            table.add_row(
                str(rank),
                str(row["model"]),
                str(row["cases"]),
                f"{row['completed']}/{row['cases']}",
                str(row["hard_failures"]),
                _display_score(row["overall"]),
            )
        console.print(table)
        breakdown = Table(title="Mean category scores", header_style="bold cyan")
        breakdown.add_column("Category")
        for row in models:
            breakdown.add_column(str(row["model"]), justify="right")
        for category in categories:
            breakdown.add_row(
                category.label,
                *[
                    _display_score(row.get("scores", {}).get(category.name))
                    for row in models
                ],
            )
        console.print(breakdown)
        return

    table = Table(title="ChGK translation benchmark", header_style="bold cyan")
    table.add_column("Rank", justify="right", style="dim")
    table.add_column("Model", no_wrap=True)
    table.add_column("N", justify="right")
    table.add_column("Complete", justify="right")
    for category in categories:
        table.add_column(category.label, justify="right")
    table.add_column("Hard fails", justify="right")
    table.add_column("Overall", justify="right", style="bold")
    for rank, row in enumerate(models, start=1):
        scores = row.get("scores", {})
        table.add_row(
            str(rank),
            str(row["model"]),
            str(row["cases"]),
            f"{row['completed']}/{row['cases']}",
            *[_display_score(scores.get(category.name)) for category in categories],
            str(row["hard_failures"]),
            _display_score(row["overall"]),
        )
    console.print(table)


def write_summary(summary: dict[str, object], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = [
        CategorySpec.model_validate(value) for value in summary.get("categories", [])
    ]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    headers = [
        "rank",
        "model",
        "cases",
        "completed",
        "scoring_errors",
        *[category.name for category in categories],
        "hard_failures",
        "overall",
    ]
    csv_rows: list[dict[str, object]] = []
    markdown_rows: list[list[str]] = []
    for rank, row in enumerate(summary.get("models", []), start=1):
        scores = row.get("scores", {})
        csv_row = {
            "rank": rank,
            "model": row["model"],
            "cases": row["cases"],
            "completed": row["completed"],
            "scoring_errors": row["scoring_errors"],
            **{category.name: scores.get(category.name) for category in categories},
            "hard_failures": row["hard_failures"],
            "overall": row["overall"],
        }
        csv_rows.append(csv_row)
        markdown_rows.append(
            [
                str(rank),
                str(row["model"]),
                str(row["cases"]),
                f"{row['completed']}/{row['cases']}",
                *[_display_score(scores.get(category.name)) for category in categories],
                str(row["hard_failures"]),
                _display_score(row["overall"]),
            ]
        )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(csv_rows)
    markdown_headers = [
        "Rank",
        "Model",
        "N",
        "Complete",
        *[category.label for category in categories],
        "Hard fails",
        "Overall",
    ]
    lines = [
        "| " + " | ".join(markdown_headers) + " |",
        "| " + " | ".join("---" for _ in markdown_headers) + " |",
        *("| " + " | ".join(row) + " |" for row in markdown_rows),
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_report(paths: list[str | Path], output_dir: str | Path) -> dict[str, object]:
    summary = build_summary(paths)
    write_summary(summary, output_dir)
    print_summary(summary)
    return summary
