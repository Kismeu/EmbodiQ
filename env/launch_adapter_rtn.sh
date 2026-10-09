#!/usr/bin/env bash
# Stage B second half: RTN-INT4 at both scopes. VLA-Adapter reports LLM-only and
# full-scope as CO-PRIMARY, because LLM-only covers just 24.4% of its parameters
# and yields 1.22x whole-model compression -- not a defensible headline on its own.
set -euo pipefail
source "$(dirname "$0")/common.sh"
STAMP="${1:?usage: launch_adapter_rtn.sh <stamp>}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENVS/embodiq-adapter"
for quant in rtn-llm rtn-all; do
  RUN="${quant}_${STAMP}"
  OUT="$EMB_ROOT/embodiq/results/$RUN"; mkdir -p "$OUT/logs"
  echo "=== $RUN ==="
  i=0
  for suite in libero_10 libero_goal libero_object libero_spatial; do
    CUDA_VISIBLE_DEVICES=$i nohup python -m embodiq.entry.run_adapter \
        --suite "$suite" --gpu "$i" --quant "$quant" --trials 50 \
        --out "$OUT" > "$OUT/logs/$suite.log" 2>&1 &
    i=$((i+1))
  done
  wait
  echo "=== $RUN done ==="
done
