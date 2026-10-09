#!/usr/bin/env bash
set -euo pipefail

# Install the shared embodiq package into the three already-provisioned model
# environments.  The hosts used for evaluation are often offline, so disable
# PEP 517 build isolation: setuptools is already present in each conda env and
# downloading a second build environment is unnecessary.
source "$(dirname "$0")/common.sh"

PROJECT_ROOT="${EMB_ROOT}/embodiq"
for env_name in openvla vlaadapter fastwam; do
    python_bin="${CONDA_ENVS}/${env_name}/bin/python"
    if [[ ! -x "$python_bin" ]]; then
        echo "missing environment: $python_bin" >&2
        exit 1
    fi
    "$python_bin" -m pip install -e "$PROJECT_ROOT" \
        --no-deps --no-build-isolation
    "$python_bin" -c 'import embodiq; print(embodiq.__file__)'
done

echo "embodiq deployed to: openvla, vlaadapter, fastwam"
