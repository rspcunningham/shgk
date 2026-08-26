"""Price/performance report for the model x reasoning-effort matrix.

Usage: uv run --with matplotlib python benchmarks/matrix_report.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shgk.benchmarking.models import CategorySpec
from shgk.benchmarking.report import build_summary

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "benchmarks" / "results" / "matrix"
OUTPUT = MATRIX / "price-performance.png"

# USD per 1M tokens. Anthropic first-party rates; Sonnet 5 uses its standard
# $3/$15 rather than the introductory rate, so planning numbers stay valid
# after the intro period ends.
PRICES = {
    "gpt-5.6-sol": (5.00, 0.50, 6.25, 30.00),
    "gpt-5.6-terra": (2.00, 0.25, 2.50, 12.00),
    "gpt-5.6-luna": (0.20, 0.02, 0.25, 1.20),
    "claude-opus-5": (5.00, 0.50, 6.25, 25.00),
    "claude-sonnet-5": (3.00, 0.30, 3.75, 15.00),
    "claude-haiku-4-5": (1.00, 0.10, 1.25, 5.00),
}
SLUGS = {
    "gpt-5.6-sol": "gpt-5-6-sol",
    "gpt-5.6-terra": "gpt-5-6-terra",
    "gpt-5.6-luna": "gpt-5-6-luna",
    "claude-opus-5": "claude-opus-5",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-haiku-4-5": "claude-haiku-4-5",
}
FAMILY = {m: ("openai" if m.startswith("gpt") else "anthropic") for m in PRICES}
EFFORTS = ["none", "low", "medium", "high", "xhigh", "max"]

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE, GHOST = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#dcdbd4"
SERIES = {"openai": "#2a78d6", "anthropic": "#eb6834"}
ELIGIBLE_QUESTIONS = 293_055
CASE_COUNT = 40


def cost_per_question(raw: Path, model: str) -> float | None:
    inp, cached, write, out = PRICES[model]
    costs = []
    for line in raw.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") != "completed":
            continue
        usage = record["translation"]["workflow"]["usage"]
        c, w = usage["cached_input_tokens"], usage["cache_write_input_tokens"]
        uncached = max(0, usage["input_tokens"] - c - w)
        costs.append(
            (uncached * inp + c * cached + w * write + usage["output_tokens"] * out)
            / 1_000_000
        )
    return sum(costs) / len(costs) if costs else None


def per_judge_overall(scored: Path) -> tuple[float | None, float | None]:
    """Mean overall under two aggregations.

    ``per-category`` medians every judgement inside each category. That reduces
    per-category noise but is NOT bounded by the individual judges' totals:
    each judge is harsh on different categories, so a dimension-wise median
    picks up the lenient value on each dimension and the composite can land
    above both judges (measured: 136 cases above vs 13 below).

    ``per-judge`` scores each judgement whole, then medians those composites,
    which keeps the compromise semantics people expect from a panel.
    """

    from collections import defaultdict

    per_category, per_judge = [], []
    for line in scored.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        specs, deterministic = {}, {}
        judgements: dict[str, dict[str, float]] = defaultdict(dict)
        pooled: dict[str, list[float]] = defaultdict(list)
        for entry in record.get("scoring", []):
            for raw_spec in entry.get("category_specs", []):
                spec = CategorySpec.model_validate(raw_spec)
                specs[spec.name] = spec
            if entry.get("scorer") == "deterministic":
                deterministic.update({k: float(v) for k, v in entry.get("scores", {}).items()})
            for index, member in enumerate((entry.get("metadata") or {}).get("members") or []):
                label = f"{(member.get('metadata') or {}).get('model')}#{index}"
                for name, value in member["scores"].items():
                    judgements[label][name] = float(value)
                    pooled[name].append(float(value))

        def weighted(scores: dict[str, float]) -> float | None:
            parts = [
                (
                    max(0, min(1, (v - specs[k].minimum) / (specs[k].maximum - specs[k].minimum))),
                    specs[k].weight,
                )
                for k, v in {**scores, **deterministic}.items()
                if k in specs and specs[k].weight > 0
            ]
            total = sum(w for _, w in parts)
            return 100 * sum(v * w for v, w in parts) / total if total else None

        if pooled:
            value = weighted({k: statistics.median(v) for k, v in pooled.items()})
            if value is not None:
                per_category.append(value)
        composites = [weighted(s) for s in judgements.values()]
        composites = [c for c in composites if c is not None]
        if composites:
            per_judge.append(statistics.median(composites))
    return (
        statistics.mean(per_category) if per_category else None,
        statistics.mean(per_judge) if per_judge else None,
    )


def per_case_overall(scored: Path) -> dict[str, float]:
    out = {}
    for line in scored.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        specs, merged = {}, {}
        for entry in record.get("scoring", []):
            for raw_spec in entry.get("category_specs", []):
                spec = CategorySpec.model_validate(raw_spec)
                specs[spec.name] = spec
            merged.update({k: float(v) for k, v in entry.get("scores", {}).items()})
        weighted = [
            (
                max(0, min(1, (v - specs[k].minimum) / (specs[k].maximum - specs[k].minimum))),
                specs[k].weight,
            )
            for k, v in merged.items()
            if k in specs and specs[k].weight > 0
        ]
        total = sum(w for _, w in weighted)
        if total:
            out[record["case_id"]] = 100 * sum(v * w for v, w in weighted) / total
    return out


def collect() -> list[dict]:
    points = []
    for model, slug in SLUGS.items():
        for effort in EFFORTS:
            raw, scored = MATRIX / f"{slug}-{effort}.raw.jsonl", MATRIX / f"{slug}-{effort}.scored.jsonl"
            if not (raw.is_file() and scored.is_file()):
                continue
            summary = build_summary([scored])["models"]
            if not summary or summary[0]["overall"] is None:
                continue
            row = summary[0]
            cases = per_case_overall(scored)
            by_category, by_judge = per_judge_overall(scored)
            points.append(
                {
                    "model": model,
                    "effort": effort,
                    "overall": row["overall"],
                    "by_judge": by_judge,
                    "hard_failures": row["hard_failures"],
                    "n": len(cases),
                    "sem": (
                        statistics.stdev(cases.values()) / len(cases) ** 0.5
                        if len(cases) > 1
                        else 0.0
                    ),
                    "cost": cost_per_question(raw, model),
                }
            )
    return points


def plot(points: list[dict]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=200, sharex=True, sharey=True)
    figure.patch.set_facecolor(SURFACE)
    all_xy = [(p["cost"], p["overall"]) for p in points if p["cost"]]

    for index, model in enumerate(SLUGS):
        ax = axes[index // 3][index % 3]
        ax.set_facecolor(SURFACE)
        ax.scatter(*zip(*all_xy), s=26, color=GHOST, zorder=1)  # every config, for context
        series = sorted(
            (p for p in points if p["model"] == model),
            key=lambda p: EFFORTS.index(p["effort"]),
        )
        color = SERIES[FAMILY[model]]
        if series:
            xs = [p["cost"] for p in series]
            ys = [p["overall"] for p in series]
            ax.plot(xs, ys, color=color, linewidth=2, zorder=2)
            ax.scatter(xs, ys, s=80, color=color, edgecolors=SURFACE, linewidths=1.8, zorder=3)
            for p in series:
                ax.annotate(
                    p["effort"],
                    (p["cost"], p["overall"]),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=INK2,
                )
        ax.set_title(model, color=INK, fontsize=11, loc="left", pad=8)
        ax.set_xscale("log")
        ax.grid(True, color=GRID, linewidth=0.7, zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=9)

    figure.suptitle(
        "ChGK translation: price vs performance, 6 models x reasoning effort",
        color=INK,
        fontsize=14,
        x=0.5,
        y=0.98,
    )
    figure.text(
        0.5,
        0.94,
        "40 fixed cases | writer=critic=editor | judged by a gpt-5.6-sol + claude-sonnet-5 panel "
        "(2 passes each, per-category median) | grey = all configs",
        ha="center",
        fontsize=9,
        color=INK2,
    )
    figure.text(0.5, 0.02, "Measured cost per question (USD, log scale)", ha="center", color=INK2)
    figure.text(
        0.02, 0.5, "Overall rubric score (weighted, 0-100)", va="center",
        rotation="vertical", color=INK2,
    )
    figure.tight_layout(rect=(0.035, 0.04, 1, 0.93))
    figure.savefig(OUTPUT, facecolor=SURFACE)
    print(f"\nwrote {OUTPUT}")


def main() -> None:
    collected = collect()
    # A config scored on fewer than all 40 cases is not comparable with a
    # complete one — an easy subset inflates it. Report it, don't rank it.
    points = [p for p in collected if p["n"] == CASE_COUNT]
    partial = [p for p in collected if p["n"] != CASE_COUNT]
    if partial:
        print("incomplete configs (excluded from ranking and chart):")
        for p in sorted(partial, key=lambda p: p["model"]):
            print(f"  {p['model']:<18}{p['effort']:>7}  {p['n']}/{CASE_COUNT} cases")
        print()
    print(
        f"{'model':<18}{'effort':>7}{'overall':>9}{'+/-':>6}{'byJudge':>9}"
        f"{'HF':>4}{'cost/q':>10}{'corpus':>10}"
    )
    for p in sorted(points, key=lambda p: -p["overall"]):
        corpus = p["cost"] * ELIGIBLE_QUESTIONS if p["cost"] else 0
        by_judge = f"{p['by_judge']:.2f}" if p["by_judge"] is not None else "-"
        print(
            f"{p['model']:<18}{p['effort']:>7}{p['overall']:>9.2f}{p['sem']:>6.1f}"
            f"{by_judge:>9}{p['hard_failures']:>4}{p['cost']:>10.4f}{corpus:>9.0f}$"
        )
    print(f"\n{len(points)}/36 configs complete and scored")
    plot(points)


if __name__ == "__main__":
    main()
