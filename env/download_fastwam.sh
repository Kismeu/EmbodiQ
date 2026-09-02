#!/usr/bin/env bash
# Stage A, hour 0: FastWAM weight acquisition (unbounded external risk -> start first).
# Default source in the repo is ModelScope with MD5 gates; we force HuggingFace.
set -euo pipefail
export HF_HOME=${HF_HOME:-/data/jyc/hf}
CKPT_DIR="/data/jyc/space/Embodied AI/FastWAM/checkpoints/fastwam_release"
mkdir -p "$CKPT_DIR"

echo "[$(date +%T)] FastWAM release checkpoints -> $CKPT_DIR"
hf download yuanty/fastwam \
  libero_uncond_2cam224.pt \
  libero_uncond_2cam224_dataset_stats.json \
  libero_optional_idm_2cam224.pt \
  libero_optional_idm_2cam224_dataset_stats.json \
  --local-dir "$CKPT_DIR"

echo "[$(date +%T)] done"
ls -la "$CKPT_DIR"
