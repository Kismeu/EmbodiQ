#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
PREFIX="$CONDA_ENVS/embodiq-fastwam"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda create -p "$PREFIX" python=3.10 -y
conda activate "$PREFIX"
pip install -U pip setuptools wheel

# --index-url, NOT the README's --extra-index-url: with the latter pip can resolve
# the plain 2.7.1 from PyPI, and then `pip install -e .` fails its ==2.7.1+cu128 pin.
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# --no-deps then explicit deps: torchcodec and deepspeed are training-only and
# deepspeed costs a ~10 min compile for nothing.
pip install -e "$EMB_ROOT/FastWAM" --no-deps
pip install accelerate==1.12.0 transformers==4.49.0 "numpy==2.2.6" hydra-core==1.3.2 \
    omegaconf==2.3.0 safetensors==0.5.3 einops imageio "imageio[ffmpeg]" av==15.1.0 \
    modelscope==1.34.0 huggingface-hub==0.29.2 pillow rich termcolor tqdm pandas pyyaml

# matplotlib is an undeclared LIBERO import; opencv pinned because its current
# release requires numpy>=2 in the Prismatic stacks (kept consistent here).
pip install robosuite==1.4.1 mujoco==3.3.2 bddl easydict cloudpickle gym matplotlib
# NOT `pip install -e LIBERO`: upstream does not track libero/__init__.py, so
# find_packages() returns nothing and the editable install is a silent no-op.
bash "$(dirname "$0")/setup_libero.sh" --link "$PREFIX"
pip install -e "$EMB_ROOT/embodiq[quant,nvml]"

python - <<'PY'
import torch, transformers, numpy
assert torch.__version__.startswith("2.7"), torch.__version__
assert transformers.__version__ == "4.49.0", transformers.__version__
assert numpy.__version__.startswith("2.2"), numpy.__version__
assert hasattr(torch.ops.aten, "_weight_int4pack_mm")
print("version gates OK:", torch.__version__, transformers.__version__, numpy.__version__)
PY
echo "built $PREFIX"
