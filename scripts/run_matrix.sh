#!/usr/bin/env bash
# Model x reasoning-effort matrix, scored by the sol+sonnet panel on rubric v2.
# Resumable: existing raw/scored files are reused, so re-running finishes gaps.
#
#   scripts/run_matrix.sh gen   <provider:model>   # generation only (one model, all efforts)
#   scripts/run_matrix.sh score <provider:model>   # panel scoring only
#
# Generation streams run one per model in parallel. Scoring is deliberately kept
# separate: every stream shares the same two judge models, so scoring all models
# at once would stack hundreds of concurrent judge calls on one pair of APIs.
set -u
phase="${1:?usage: run_matrix.sh gen|score <provider:model>}"
spec="${2:?usage: run_matrix.sh gen|score <provider:model>}"
outdir="benchmarks/results/matrix"
efforts=(none low medium high xhigh max)
gen_concurrency="${GEN_CONCURRENCY:-20}"
score_concurrency="${SCORE_CONCURRENCY:-8}"

provider="${spec%%:*}"; model="${spec#*:}"
slug=$(printf '%s' "$model" | tr 'A-Z' 'a-z' | sed 's/[^a-z0-9]/-/g; s/--*/-/g; s/^-//; s/-$//')
mkdir -p "$outdir"
failures=0

for effort in "${efforts[@]}"; do
  raw="$outdir/${slug}-${effort}.raw.jsonl"
  scored="$outdir/${slug}-${effort}.scored.jsonl"
  case "$phase" in
    gen)
      echo "=== GEN $spec @ $effort ==="
      uv run shgk benchmark --provider "$provider" --model "$model" \
        --cases benchmarks/cases.jsonl --output "$raw" \
        --reasoning-effort "$effort" --concurrency "$gen_concurrency" \
        || { echo "!! GEN FAILED $spec @ $effort"; failures=$((failures+1)); }
      ;;
    score)
      [ -s "$raw" ] || { echo "-- skip score $spec @ $effort (no raw)"; continue; }
      echo "=== SCORE $spec @ $effort ==="
      uv run shgk benchmark-score --input "$raw" --output "$scored" \
        --concurrency "$score_concurrency" \
        || { echo "!! SCORE FAILED $spec @ $effort"; failures=$((failures+1)); }
      ;;
    *) echo "unknown phase $phase" >&2; exit 2 ;;
  esac
done
echo "MATRIX $phase $spec COMPLETE failures=$failures"
exit $failures
