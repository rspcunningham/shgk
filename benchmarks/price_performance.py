"""Price/performance comparison of model x reasoning-effort benchmark runs.

Usage: uv run --with matplotlib python benchmarks/price_performance.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shgk.benchmarking.report import build_summary

ROOT = Path(__file__).resolve().parent.parent
AB_DIR = ROOT / "benchmarks" / "results" / "effort-ab"
SUITE_DIR = ROOT / "benchmark"
OUTPUT = AB_DIR / "price-performance.png"

# USD per 1M tokens (OpenAI, August 2026; cache write = 1.25x input).
PRICES = {
    "gpt-5.6-sol": {"input": 5.00, "cached": 0.50, "write": 6.25, "output": 30.00},
    "gpt-5.6-terra": {"input": 2.00, "cached": 0.25, "write": 2.50, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "cached": 0.02, "write": 0.25, "output": 1.20},
}

RUNS = {
    "gpt-5.6-sol": {
        "none": AB_DIR / "sol-none",
        "low": SUITE_DIR / "openai-gpt-5-6-sol",
        "medium": AB_DIR / "sol-medium",
        "high": AB_DIR / "sol-high",
    },
    "gpt-5.6-terra": {
        "low": AB_DIR / "terra-low",
        "medium": AB_DIR / "terra-medium",
        "high": AB_DIR / "terra-high",
    },
    "gpt-5.6-luna": {
        "none": AB_DIR / "luna-none",
        "low": SUITE_DIR / "openai-gpt-5-6-luna",
        "medium": AB_DIR / "luna-medium",
        "high": AB_DIR / "luna-high",
        "xhigh": AB_DIR / "luna-xhigh",
        "max": AB_DIR / "luna-max",
    },
}
EFFORT_ORDER = ["none", "low", "medium", "high", "xhigh", "max"]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = {
    "gpt-5.6-sol": "#2a78d6",
    "gpt-5.6-luna": "#eb6834",
    "gpt-5.6-terra": "#1baf7a",
}


def mean_cost_per_question(raw_path: Path, model: str) -> float:
    price = PRICES[model]
    costs = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("status") != "completed":
            continue
        usage = record["translation"]["workflow"]["usage"]
        cached = usage["cached_input_tokens"]
        write = usage["cache_write_input_tokens"]
        uncached = max(0, usage["input_tokens"] - cached - write)
        costs.append(
            (
                uncached * price["input"]
                + cached * price["cached"]
                + write * price["write"]
                + usage["output_tokens"] * price["output"]
            )
            / 1_000_000
        )
    return sum(costs) / len(costs)


def collect() -> list[dict[str, object]]:
    points = []
    for model, efforts in RUNS.items():
        for effort in EFFORT_ORDER:
            if effort not in efforts:
                continue
            stem = efforts[effort]
            summary = build_summary([stem.with_suffix(".scored.jsonl")])
            row = summary["models"][0]
            points.append(
                {
                    "model": model,
                    "effort": effort,
                    "overall": row["overall"],
                    "hard_failures": row["hard_failures"],
                    "completed": f"{row['completed']}/{row['cases']}",
                    "cost": mean_cost_per_question(stem.with_suffix(".raw.jsonl"), model),
                }
            )
    return points


def plot(points: list[dict[str, object]]) -> None:
    figure, axes = plt.subplots(figsize=(9.6, 6.0), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    axes.set_facecolor(SURFACE)

    for model in RUNS:
        series = [p for p in points if p["model"] == model]
        xs = [p["cost"] for p in series]
        ys = [p["overall"] for p in series]
        color = SERIES[model]
        axes.plot(xs, ys, color=color, linewidth=2, zorder=2, label=model)
        axes.scatter(
            xs, ys, s=90, color=color, edgecolors=SURFACE, linewidths=2, zorder=3
        )
        offsets = {
            ("gpt-5.6-sol", "medium"): ((0, 12), (-22, 4)),
            ("gpt-5.6-sol", "high"): ((26, 0), (26, -12)),
            ("gpt-5.6-sol", "low"): ((14, 8), (16, -16)),
            ("gpt-5.6-sol", "none"): ((-20, 6), (-18, -16)),
            ("gpt-5.6-luna", "low"): ((-18, 8), (-20, -16)),
            ("gpt-5.6-luna", "medium"): ((30, -4), (32, -16)),
            ("gpt-5.6-luna", "high"): ((0, 12), (0, -18)),
            ("gpt-5.6-luna", "none"): ((0, 12), (0, -18)),
        }
        for p in series:
            effort_offset, failure_offset = offsets.get(
                (model, p["effort"]), ((0, 12), (0, -18))
            )
            axes.annotate(
                f"{p['effort']}",
                (p["cost"], p["overall"]),
                xytext=effort_offset,
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=INK_SECONDARY,
            )
            axes.annotate(
                f"{p['hard_failures']} HF",
                (p["cost"], p["overall"]),
                xytext=failure_offset,
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=INK_MUTED,
            )

    axes.set_xscale("log")
    ticks = [0.003, 0.01, 0.03, 0.1, 0.3]
    axes.set_xticks(ticks)
    axes.set_xticklabels(
        [f"${t:.3f}".rstrip("0").rstrip(".") if t < 0.01 else f"${t:g}" for t in ticks]
    )
    axes.minorticks_off()
    axes.set_xlabel("Measured cost per question (USD, log scale)", color=INK_SECONDARY)
    axes.set_ylabel("Overall rubric score (weighted, 0-100)", color=INK_SECONDARY)
    axes.set_title(
        "ChGK translation: price vs performance by reasoning effort",
        color=INK,
        fontsize=13,
        pad=28,
    )
    axes.text(
        0,
        1.03,
        "40 fixed cases, writer=critic=editor, judged by gpt-5.6-sol; "
        "labels: reasoning effort, HF = hard failures",
        transform=axes.transAxes,
        fontsize=9,
        color=INK_SECONDARY,
    )
    axes.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        axes.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color(BASELINE)
    axes.tick_params(colors=INK_MUTED)
    legend = axes.legend(loc="lower right", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK)

    figure.tight_layout()
    figure.savefig(OUTPUT, facecolor=SURFACE)
    print(f"wrote {OUTPUT}")


def main() -> None:
    points = collect()
    for p in sorted(points, key=lambda p: -(p["overall"] or 0)):
        print(
            f"{p['model']:>13} {p['effort']:>7}: overall={p['overall']:.2f} "
            f"hard_fails={p['hard_failures']:>2} complete={p['completed']} "
            f"cost/question=${p['cost']:.4f}"
        )
    plot(points)


if __name__ == "__main__":
    main()
