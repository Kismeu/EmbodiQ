#!/usr/bin/env bash
# VLA-Adapter LIBERO checkpoints (the pilot model).
#
# Downloaded to a directory OUTSIDE the repo, then frozen: get_vla() rewrites
# config.json and copies modeling_prismatic.py into the checkpoint dir on every
# load, so four concurrent workers would race on a 3 GB directory. We do that
# sync once, hash it, and make it read-only.
set -euo pipefail
source "$(dirname "$0")/common.sh"
DEST="${EMB_ROOT}/checkpoints/vla-adapter"
mkdir -p "$DEST"
for suite in Spatial Object Goal Long; do
  d="$DEST/LIBERO-${suite}-Pro"
  if [ -f "$d/model.safetensors" ]; then echo "have $suite"; continue; fi
  echo "[$(date +%T)] VLA-Adapter/LIBERO-${suite}-Pro"
  hf download "VLA-Adapter/LIBERO-${suite}-Pro" --local-dir "$d"
done
du -sh "$DEST"/* 2>/dev/null
