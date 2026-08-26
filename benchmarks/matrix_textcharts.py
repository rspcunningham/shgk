"""Terminal renderings of the matrix charts."""
from __future__ import annotations
from matrix_report import CASE_COUNT, ELIGIBLE_QUESTIONS, FAMILY, collect

SHORT = {"gpt-5.6-sol":"sol","gpt-5.6-terra":"terra","gpt-5.6-luna":"luna",
         "claude-opus-5":"opus-5","claude-sonnet-5":"sonnet-5","claude-haiku-4-5":"haiku-4.5"}
def name(p): return f"{SHORT[p['model']]}@{p['effort']}"

points = [p for p in collect() if p["n"] == CASE_COUNT and p["cost"]]

# ---------- 1. Pareto ----------
print("\n\033[1m1. PRICE vs QUALITY\033[0m   (x = cost/question, log scale)\n")
lo, hi = min(p["cost"] for p in points), max(p["cost"] for p in points)
import math
W = 62
def col(c): return int((math.log10(c)-math.log10(lo))/(math.log10(hi)-math.log10(lo))*(W-1))
frontier, best = set(), -1
for p in sorted(points, key=lambda p: p["cost"]):
    if p["overall"] > best: frontier.add(id(p)); best = p["overall"]
rows = {}
for p in points:
    band = round(p["overall"])
    rows.setdefault(band, [None]*W)
    ch = "\033[1m@\033[0m" if id(p) in frontier else ("o" if FAMILY[p["model"]]=="openai" else "*")
    rows[band][col(p["cost"])] = ch
for band in sorted(rows, reverse=True):
    line = "".join(c if c else " " for c in rows[band])
    print(f"  {band:>3} |{line}")
print(f"      +{'-'*W}")
print(f"       {'$0.003':<20}{'$0.01':<20}{'$0.1':<15}$0.2")
print("       \033[1m@\033[0m = on the efficient frontier   o = OpenAI   * = Anthropic")
print("       frontier:", ", ".join(name(p) for p in sorted((p for p in points if id(p) in frontier), key=lambda p: p['cost'])))

# ---------- 2. Significance ----------
print("\n\033[1m2. SCORE with 95% CONFIDENCE INTERVALS\033[0m   (why the ranking is mostly noise)\n")
ranked = sorted(points, key=lambda p: -p["overall"])
leader = ranked[0]; cutoff = leader["overall"] - 1.96*leader["sem"]
smin = min(p["overall"]-1.96*p["sem"] for p in points)
smax = max(p["overall"]+1.96*p["sem"] for p in points)
BW = 54
def sc(v): return max(0, min(BW-1, int((v-smin)/(smax-smin)*(BW-1))))
tied = 0
for p in ranked:
    a, b, m = sc(p["overall"]-1.96*p["sem"]), sc(p["overall"]+1.96*p["sem"]), sc(p["overall"])
    bar = list(" "*BW)
    for i in range(a, b+1): bar[i] = "─"
    bar[a], bar[b], bar[m] = "├", "┤", "●"
    overlaps = p["overall"]+1.96*p["sem"] >= cutoff
    tied += overlaps
    mark = "\033[1m" if overlaps else "\033[2m"
    print(f"  {name(p):<17}{p['overall']:6.2f}  {mark}{''.join(bar)}\033[0m")
print(f"\n  {'':<17}{'':<6}  {smin:.0f}{' '*(BW-8)}{smax:.0f}")
print(f"  \033[1mbold\033[0m = CI reaches the leader's CI floor ({cutoff:.1f}) -> not distinguishable from #1: {tied}/{len(points)} configs")

# ---------- 3. Hard failures ----------
print("\n\033[1m3. HARD FAILURES\033[0m   (integrity violations in 40 cases - the metric that separates)\n")
for p in sorted(points, key=lambda p: (p["hard_failures"], p["cost"]))[:12]:
    bar = "█"*p["hard_failures"]
    print(f"  {name(p):<17}{bar:<20}{p['hard_failures']:>3}   ${p['cost']:.4f}/q")
print("  ...")
for p in sorted(points, key=lambda p: (-p["hard_failures"], p["cost"]))[:3]:
    print(f"  {name(p):<17}{'█'*p['hard_failures']:<20}{p['hard_failures']:>3}   ${p['cost']:.4f}/q")

# ---------- 4. Corpus cost ----------
print(f"\n\033[1m4. COST TO TRANSLATE ALL {ELIGIBLE_QUESTIONS:,} QUESTIONS\033[0m   (configs with <=5 hard failures)\n")
short = sorted((p for p in points if p["hard_failures"] <= 5), key=lambda p: p["cost"])[:10]
mx = max(p["cost"]*ELIGIBLE_QUESTIONS for p in short)
for p in short:
    total = p["cost"]*ELIGIBLE_QUESTIONS
    bar = "█"*max(1, int(total/mx*40))
    print(f"  {name(p):<17}{bar:<42}${total/1000:>6.1f}k   ({p['overall']:.1f}, {p['hard_failures']} HF)")
print()
