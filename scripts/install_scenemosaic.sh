#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAM3D_DIR="$REPO_ROOT/sam-3d-objects"

# pip package indexes used for dependency resolution; override them (e.g. a
# mirror or an internal proxy) before running this script. The defaults keep
# the original upstream behavior.
export SCENEMOSAIC_PIP_INDEX="${SCENEMOSAIC_PIP_INDEX:-https://pypi.org/simple}"
export SCENEMOSAIC_PIP_EXTRA_INDEX="${SCENEMOSAIC_PIP_EXTRA_INDEX:-https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121}"
export SCENEMOSAIC_KAOLIN_FIND_LINKS="${SCENEMOSAIC_KAOLIN_FIND_LINKS:-https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html}"

ENV_NAME="$(printenv SCENEMOSAIC_ENV_NAME || true)"
ENV_PREFIX="$(printenv SCENEMOSAIC_ENV_PREFIX || true)"
if [[ -z "$ENV_NAME" && -z "$ENV_PREFIX" ]]; then
  ENV_NAME="scenemosaic"
elif [[ -n "$ENV_NAME" && -n "$ENV_PREFIX" ]]; then
  echo "Set only one of SCENEMOSAIC_ENV_NAME or SCENEMOSAIC_ENV_PREFIX" >&2
  exit 1
fi

if command -v mamba >/dev/null 2>&1; then
  ENV_MANAGER=mamba
elif command -v conda >/dev/null 2>&1; then
  ENV_MANAGER=conda
else
  echo "mamba or conda is required" >&2
  exit 1
fi

if [[ ! -f "$SAM3D_DIR/environments/default.yml" ]]; then
  echo "SAM 3D Objects source not found at $SAM3D_DIR" >&2
  exit 1
fi

if [[ -n "$ENV_PREFIX" ]]; then
  ENV_LABEL="$ENV_PREFIX"
  if [[ -x "$ENV_PREFIX/bin/python" ]]; then
    echo "Using existing environment prefix: $ENV_PREFIX"
  else
    "$ENV_MANAGER" env create       --prefix "$ENV_PREFIX"       --file "$SAM3D_DIR/environments/default.yml"
  fi
else
  ENV_LABEL="$ENV_NAME"
  if "$ENV_MANAGER" env list     | awk 'NF && $1 !~ /^#/ {print $1}'     | grep -Fxq "$ENV_NAME"; then
    echo "Using existing environment: $ENV_NAME"
  else
    "$ENV_MANAGER" env create       --name "$ENV_NAME"       --file "$SAM3D_DIR/environments/default.yml"
  fi
fi

run_in_env() {
  if [[ -n "$ENV_PREFIX" ]]; then
    "$ENV_MANAGER" run --prefix "$ENV_PREFIX" "$@"
  else
    "$ENV_MANAGER" run --name "$ENV_NAME" "$@"
  fi
}

PIP_EXTRA_INDEX_URL="$SCENEMOSAIC_PIP_EXTRA_INDEX"   run_in_env python -m pip install --editable "$SAM3D_DIR[dev]"

PIP_EXTRA_INDEX_URL="$SCENEMOSAIC_PIP_EXTRA_INDEX"   run_in_env python -m pip install --editable "$SAM3D_DIR[p3d]"

PIP_EXTRA_INDEX_URL="$SCENEMOSAIC_PIP_EXTRA_INDEX" PIP_FIND_LINKS="$SCENEMOSAIC_KAOLIN_FIND_LINKS"   run_in_env python -m pip install --editable "$SAM3D_DIR[inference]"

run_in_env python "$SAM3D_DIR/patching/hydra"

run_in_env python -m pip install --requirement "$REPO_ROOT/requirements.txt"   --index-url "$SCENEMOSAIC_PIP_INDEX"

for MODULE_NAME in   OpenGL coacd fast_simplification open_clip openai pybullet sapien shapely; do
  run_in_env python -c "import $MODULE_NAME"
done
echo "SceneMosaic dependencies verified"

if ! PYOPENGL_PLATFORM=egl run_in_env python -c 'import ctypes, ctypes.util; library = ctypes.util.find_library("EGL"); assert library, "system dynamic linker cannot find EGL"; ctypes.CDLL(library); import pyrender'; then
  echo "SceneMosaic requires a system EGL runtime for headless rendering." >&2
  echo "On Ubuntu/Debian install: sudo apt-get install libegl1 libglvnd0 libegl-mesa0" >&2
  exit 1
fi
echo "SceneMosaic EGL runtime verified"

echo "SceneMosaic environment ready: $ENV_LABEL"
run_in_env python -c 'import sys; print(sys.executable)'
