#!/usr/bin/env bash
# Shared environment for every embodiq worker. Source this, never edit a repo.
export EMB_ROOT="/data/jyc/space/Embodied AI"
export CONDA_ROOT="/data/jyc/tools/anaconda3"
export CONDA_ENVS="${CONDA_ROOT}/envs"
export HF_HOME="${HF_HOME:-/data/jyc/hf}"

# Headless offscreen rendering. libEGL_nvidia.so + 10_nvidia.json verified present,
# no DISPLAY. Falling back to osmesa costs ~4x on render and adds ~10 h to the matrix.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# TensorFlow 2.15 is on the eval path of both Prismatic repos purely for an image
# resize, yet it seizes the whole GPU on first use. That destroys the peak-VRAM
# measurement and blocks a second worker per GPU. embodiq.backends.compat.tf_cpu_guard
# enforces this in-process; the env vars are belt-and-braces.
export QVLA_TF_CPU_ONLY=1
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# FastWAM's loader defaults to ModelScope with MD5 gates.
export DIFFSYNTH_DOWNLOAD_SOURCE=huggingface
export DIFFSYNTH_MODEL_BASE_PATH="${EMB_ROOT}/FastWAM/checkpoints"
