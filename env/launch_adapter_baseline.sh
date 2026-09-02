#!/usr/bin/env bash
# One suite per GPU: VLA-Adapter ships a different checkpoint per suite, so a
# worker that hopped suites would reload a 3 GB model for nothing. Four suites
# across four GPUs is naturally balanced; libero_10 (520 max_steps) is the
# critical path at roughly 2x the others.
set -euo pipefail
source "$(dirname "$0")/common.sh"
RUN_ID="${1:?usage: launch_adapter_baseline.sh <run_id> [quant] [trials]}"
QUANT="${2:-fp16}"
TRIALS="${3:-50}"
OUT="$EMB_ROOT/embodiq/results/$RUN_ID"
LOGS="$EMB_ROOT/embodiq/results/$RUN_ID/logs"
mkdir -p "$LOGS"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENVS/embodiq-adapter"

i=0
for suite in libero_10 libero_goal libero_object libero_spatial; do
  CUDA_VISIBLE_DEVICES=$i nohup python -m embodiq.entry.run_adapter \
      --suite "$suite" --gpu "$i" --quant "$QUANT" --trials "$TRIALS" \
      --out "$OUT" > "$LOGS/$suite.log" 2>&1 &
  echo "  gpu$i <- $suite (pid $!)"
  i=$((i+1))
done
echo "run_id=$RUN_ID  out=$OUT"
wait
