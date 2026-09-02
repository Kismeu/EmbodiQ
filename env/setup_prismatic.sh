#!/usr/bin/env bash
# Builds ONE of the two Prismatic-derived environments.
#   usage: bash env/setup_prismatic.sh openvla | adapter
#
# Two prefixes, not one, even though the pins are identical: openvla/pyproject.toml
# and VLA-Adapter/pyproject.toml both use packages.find where=["."], so both
# editable-install top-level packages named prismatic / experiments / vla-scripts.
# Sharing a prefix silently import-shadows one repo with the other.
set -euo pipefail
source "$(dirname "$0")/common.sh"

WHICH="${1:?usage: setup_prismatic.sh openvla|adapter}"
case "$WHICH" in
  openvla) REPO="$EMB_ROOT/openvla" ;;
  adapter) REPO="$EMB_ROOT/VLA-Adapter" ;;
  *) echo "unknown target: $WHICH" >&2; exit 2 ;;
esac
PREFIX="$CONDA_ENVS/embodiq-$WHICH"

source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda create -p "$PREFIX" python=3.10 -y
conda activate "$PREFIX"

pip install -U pip setuptools wheel packaging ninja

# torch first, from the pytorch index. cu121 wheels run fine on the 12.6 driver.
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# numpy pinned BEFORE tensorflow/robosuite, both of which will otherwise move it.
pip install "numpy==1.26.4"

pip install -e "$REPO"

# --- dependency repairs -------------------------------------------------
# Neither repo pins its full transitive set, and pip's resolver cannot reach the
# combination that actually works. VLA-Adapter/our_envs.txt -- the authors'
# frozen `pip list` -- is the ground truth, so we match it directly with
# --no-deps rather than fighting the resolver.
#
#   * tensorflow-datasets pulls tensorflow-metadata unpinned; current releases
#     need protobuf>=5.27 while tensorflow 2.15 caps it below 5. Symptom:
#     "cannot import name runtime_version" when importing prismatic.
#   * current wandb ships protobuf-5 stubs. Symptom: "cannot import name
#     Imports from wandb.proto.wandb_telemetry_pb2".
pip install --no-deps "tensorflow-metadata==1.16.1" "protobuf==3.20.3" "wandb==0.19.8"
pip install --no-deps docker-pycreds sentry-sdk setproctitle eval-type-backport

#   * robot_utils.py imports msgpack at MODULE scope (line ~206) for the ALOHA
#     client, so the LIBERO eval cannot even be imported without it.
pip install --no-deps msgpack msgpack-numpy
pip install requests
# pip regenerates the TRACKED <pkg>.egg-info/, dirtying the upstream repo.
bash "$(dirname "$0")/protect_upstream.sh"

# Install the prebuilt wheel by URL rather than `pip install flash-attn`.
# flash-attn's setup.py DOES locate this exact wheel, but then os.rename()s it
# from its /tmp build dir into the /data pip cache and dies with
# "Errno 18 Invalid cross-device link". Fetching it directly skips setup.py,
# avoids a ~40 min source compile, and is deterministic.
# cxx11abiFALSE matches the download.pytorch.org torch wheels (_GLIBCXX_USE_CXX11_ABI=0).
FA_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
pip install --no-deps "$FA_WHL"

# mujoco 3.3.2 (not 3.3.0) so both stacks share one physics version; FastWAM's
# README requires mujoco to match its LIBERO data version. Regression-verify the
# FP16 baselines after this choice.
# opencv-python is an UNPINNED robosuite dep and its current release requires
# numpy>=2, which silently drags the whole env off the 1.26.4 pin.
pip install robosuite==1.4.1 mujoco==3.3.2 "opencv-python==4.10.0.84" \
    bddl easydict imageio "imageio[ffmpeg]" cloudpickle gym

# LIBERO's setup.py uses find_packages(), but upstream does not track
# libero/__init__.py, so find_packages() returns nothing and `pip install -e`
# is a silent no-op. Put the repo root on sys.path instead and let PEP 420
# implicit namespace packages resolve libero.libero.
bash "$(dirname "$0")/setup_libero.sh" --link "$PREFIX"

# Re-pin numpy LAST: several of the above declare a bare `numpy` dependency and
# will happily upgrade it out from under the pinned stack.
pip install "numpy==1.26.4"
pip install -e "$EMB_ROOT/embodiq[quant,nvml]"

python - <<'PY'
import importlib, sys
import timm, numpy, torch, transformers
# timm is a HARD gate: modeling_prismatic raises NotImplementedError off-whitelist.
assert timm.__version__ in {"0.9.10","0.9.11","0.9.12","0.9.16"}, f"timm {timm.__version__} will hard-fail"
assert numpy.__version__.startswith("1.26"), numpy.__version__
assert torch.__version__.startswith("2.2"), torch.__version__
assert transformers.__version__ == "4.40.1", transformers.__version__
# tinygemm must exist in torch 2.2 or the whole packed-INT4 plan needs a fallback.
assert hasattr(torch.ops.aten, "_weight_int4pack_mm"), "no tinygemm in this torch"
assert hasattr(torch.ops.aten, "_convert_weight_to_int4pack")
print("version gates OK:", torch.__version__, transformers.__version__, timm.__version__, numpy.__version__)
PY
echo "built $PREFIX"
