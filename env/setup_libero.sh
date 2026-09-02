#!/usr/bin/env bash
# Shared LIBERO clone + per-env wiring.
#   bash setup_libero.sh                  # clone only
#   bash setup_libero.sh --link <prefix>  # also wire it into that conda prefix
#
# Evaluation needs only the repo's own bddl_files and init_files; the ~10 GB RLDS
# datasets are for fine-tuning and are NOT required.
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "$EMB_ROOT"
[ -d LIBERO ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git

ROOT="$EMB_ROOT/LIBERO/libero/libero"

# LIBERO prompts on stdin the first time it is imported if this file is absent,
# which hangs any non-interactive worker. Write it up front.
mkdir -p "$HOME/.libero"
if [ ! -f "$HOME/.libero/config.yaml" ]; then
  cat > "$HOME/.libero/config.yaml" <<EOF
assets: $ROOT/assets
benchmark_root: $ROOT
bddl_files: $ROOT/bddl_files
datasets: $EMB_ROOT/LIBERO/datasets
init_states: $ROOT/init_files
EOF
  echo "wrote $HOME/.libero/config.yaml"
fi

if [ "${1:-}" = "--link" ]; then
  PREFIX="${2:?--link needs a conda prefix}"
  # Ask the prefix's own interpreter. Globbing lib/python3.*/site-packages is
  # fragile: some conda prefixes carry both a python3.1 and a python3.10 dir.
  SP=$("$PREFIX/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  # A .pth, not `pip install -e`: upstream does not track libero/__init__.py, so
  # find_packages() finds nothing and the editable install installs nothing.
  echo "$EMB_ROOT/LIBERO" > "$SP/embodiq-libero.pth"
  echo "linked LIBERO into $SP"
fi
