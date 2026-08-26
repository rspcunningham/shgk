"""Charts for the model x reasoning-effort matrix.

Usage: uv run --with matplotlib python benchmarks/matrix_charts.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matrix_report import (
    ELIGIBLE_QUESTIONS,
    FAMILY,
    MATRIX,
    collect,
    CASE_COUNT,
)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE, GHOST = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#dcdbd4"
SERIES = {"openai": "#2a78d6", "anthropic": "#eb6834"}
SHORT = {
    "gpt-5.6-sol": "sol",
    "gpt-5.6-terra": "terra",
    "gpt-5.6-luna": "luna",
    "claude-opus-5": "opus-5",
    "claude-sonnet-5": "sonnet-5",
    "claude-haiku-4-5": "haiku-4.5",
}


def _frame(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=9)


def pareto_chart(points):
    """Cost vs quality, with the efficient frontier called out."""
    figure, ax = plt.subplots(figsize=(11.5, 7), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    _frame(ax)

    frontier, best = [], -1
    for p in sorted(points, key=lambda p: p["cost"]):
        if p["overall"] > best:
            frontier.append(p)
            best = p["overall"]

    ax.step(
        [p["cost"] for p in frontier],
        [p["overall"] for p in frontier],
        where="post",
        color=BASELINE,
        linewidth=2,
        zorder=1,
        label="efficient frontier",
    )
    for family in ("openai", "anthropic"):
        sel = [p for p in points if FAMILY[p["model"]] == family]
        ax.scatter(
            [p["cost"] for p in sel],
            [p["overall"] for p in sel],
            s=70,
            color=SERIES[family],
            edgecolors=SURFACE,
            linewidths=1.5,
            zorder=3,
            label=family,
        )

    offsets = {
        ("gpt-5.6-luna", "none"): (12, -14),
        ("gpt-5.6-luna", "low"): (10, -16),
        ("gpt-5.6-luna", "max"): (-14, 12),
        ("gpt-5.6-terra", "xhigh"): (-30, -20),
        ("claude-opus-5", "none"): (10, -20),
        ("claude-opus-5", "low"): (-96, 4),
        ("gpt-5.6-sol", "low"): (-18, 14),
        ("gpt-5.6-sol", "xhigh"): (12, 2),
    }
    for p in frontier:
        dx, dy = offsets.get((p["model"], p["effort"]), (10, 8))
        ax.annotate(
            f"{SHORT[p['model']]} @ {p['effort']}",
            (p["cost"], p["overall"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=9,
            color=INK,
            fontweight="medium",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Measured cost per question (USD, log scale)", color=INK2)
    ax.set_ylabel("Overall rubric score (weighted, 0-100)", color=INK2)
    ax.set_title(
        "Price vs quality: only the labelled configs are on the efficient frontier",
        color=INK, fontsize=13, loc="left", pad=26,
    )
    ax.text(
        0, 1.02,
        f"{len(points)} configs | 40 fixed cases | cross-family judge panel | "
        "everything above the step line is dominated",
        transform=ax.transAxes, fontsize=9, color=INK2,
    )
    legend = ax.legend(loc="lower right", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK)
    figure.tight_layout()
    out = MATRIX / "chart-pareto.png"
    figure.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def significance_chart(points):
    """Ranked scores with 95% CIs — shows how much of the ranking is noise."""
    ranked = sorted(points, key=lambda p: p["overall"])
    figure, ax = plt.subplots(figsize=(10, 11), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    _frame(ax)

    leader = max(points, key=lambda p: p["overall"])
    cutoff = leader["overall"] - 1.96 * leader["sem"]
    # "Tied" means the config's 95% CI overlaps the leader's, which is the
    # honest test — not merely whether its point estimate sits in the band.
    tied = sum(
        1 for p in points if p["overall"] + 1.96 * p["sem"] >= cutoff
    )
    ax.axvspan(cutoff, 100, color=GHOST, zorder=0)

    ys = range(len(ranked))
    for y, p in zip(ys, ranked):
        color = SERIES[FAMILY[p["model"]]]
        ax.errorbar(
            p["overall"], y, xerr=1.96 * p["sem"],
            fmt="o", color=color, ecolor=color, elinewidth=1.6,
            capsize=3, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=3,
        )
    ax.set_yticks(list(ys))
    ax.set_yticklabels(
        [f"{SHORT[p['model']]} @ {p['effort']}" for p in ranked], fontsize=9, color=INK2
    )
    ax.set_ylim(-1, len(ranked))
    ax.set_xlabel("Overall rubric score with 95% CI", color=INK2)
    ax.set_title(
        "Most of this ranking is noise", color=INK, fontsize=13, loc="left", pad=26
    )
    ax.text(
        0, 1.015,
        f"shaded band = the leader's 95% CI ({cutoff:.1f}+). "
        f"{tied} of {len(points)} configs have a CI reaching into it, so they are not "
        "distinguishable from the leader",
        transform=ax.transAxes, fontsize=9, color=INK2,
    )
    figure.tight_layout()
    out = MATRIX / "chart-significance.png"
    figure.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def failures_chart(points):
    """Hard failures vs cost — the discriminator that survives the noise."""
    figure, ax = plt.subplots(figsize=(11.5, 6.5), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    _frame(ax)
    for family in ("openai", "anthropic"):
        sel = [p for p in points if FAMILY[p["model"]] == family]
        ax.scatter(
            [p["cost"] for p in sel], [p["hard_failures"] for p in sel],
            s=70, color=SERIES[family], edgecolors=SURFACE, linewidths=1.5,
            zorder=3, label=family,
        )
    for p in points:
        if p["hard_failures"] <= 3:
            ax.annotate(
                f"{SHORT[p['model']]} @ {p['effort']}",
                (p["cost"], p["hard_failures"]),
                xytext=(0, 10), textcoords="offset points",
                ha="center", fontsize=8.5, color=INK,
            )
    ax.set_xscale("log")
    ax.set_xlabel("Measured cost per question (USD, log scale)", color=INK2)
    ax.set_ylabel(f"Hard failures across {CASE_COUNT} cases (lower is better)", color=INK2)
    ax.set_title(
        "Hard failures vs cost: broken puzzles, not score deltas",
        color=INK, fontsize=13, loc="left", pad=26,
    )
    ax.text(
        0, 1.02,
        "a hard failure is a concrete integrity violation (answer changed, clue added or lost, "
        "language-bound puzzle waved through); only the three best configs are labelled",
        transform=ax.transAxes, fontsize=9, color=INK2,
    )
    legend = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(INK)
    figure.tight_layout()
    out = MATRIX / "chart-hard-failures.png"
    figure.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def corpus_cost_chart(points):
    """What each shortlisted config would cost for the whole corpus."""
    shortlist = sorted(
        (p for p in points if p["hard_failures"] <= 5),
        key=lambda p: p["cost"] * ELIGIBLE_QUESTIONS,
    )[:10]
    figure, ax = plt.subplots(figsize=(11, 6), dpi=200)
    figure.patch.set_facecolor(SURFACE)
    _frame(ax)
    labels = [f"{SHORT[p['model']]} @ {p['effort']}" for p in shortlist]
    costs = [p["cost"] * ELIGIBLE_QUESTIONS / 1000 for p in shortlist]
    colors = [SERIES[FAMILY[p["model"]]] for p in shortlist]
    bars = ax.barh(range(len(shortlist)), costs, color=colors, height=0.62, zorder=3)
    for bar, p, cost in zip(bars, shortlist, costs):
        ax.annotate(
            f"${cost:,.1f}k   ({p['overall']:.1f}, {p['hard_failures']} HF)",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=9, color=INK2,
        )
    ax.set_yticks(range(len(shortlist)))
    ax.set_yticklabels(labels, fontsize=9.5, color=INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, max(costs) * 1.5)
    ax.set_xlabel(f"Projected cost to translate all {ELIGIBLE_QUESTIONS:,} eligible questions (USD thousands)", color=INK2)
    ax.set_title(
        "Corpus cost for configs with <=5 hard failures",
        color=INK, fontsize=13, loc="left", pad=26,
    )
    ax.text(
        0, 1.03,
        "extrapolated from measured per-question token usage; score and hard-failure count in brackets",
        transform=ax.transAxes, fontsize=9, color=INK2,
    )
    figure.tight_layout()
    out = MATRIX / "chart-corpus-cost.png"
    figure.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")


def main() -> None:
    points = [p for p in collect() if p["n"] == CASE_COUNT and p["cost"]]
    print(f"{len(points)} complete configs\n")
    pareto_chart(points)
    significance_chart(points)
    failures_chart(points)
    corpus_cost_chart(points)


if __name__ == "__main__":
    main()
