#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SAM3_DIR="$REPO_ROOT/sam3"

# pip package index used for PyPI dependency resolution; override it (e.g. a
# mirror or an internal proxy) before running this script. The default keeps
# the original PyPI behavior. SCENEMOSAIC_TORCH_INDEX serves the same purpose
# for the CUDA-specific PyTorch wheel index.
export SCENEMOSAIC_PIP_INDEX="${SCENEMOSAIC_PIP_INDEX:-https://pypi.org/simple}"
export SCENEMOSAIC_TORCH_INDEX="${SCENEMOSAIC_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

ENV_NAME="$(printenv SAM3_ENV_NAME || true)"
ENV_PREFIX="$(printenv SAM3_ENV_PREFIX || true)"
if [[ -z "$ENV_NAME" && -z "$ENV_PREFIX" ]]; then
  ENV_NAME="sam3"
elif [[ -n "$ENV_NAME" && -n "$ENV_PREFIX" ]]; then
  echo "Set only one of SAM3_ENV_NAME or SAM3_ENV_PREFIX" >&2
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

if [[ ! -f "$SAM3_DIR/pyproject.toml" ]]; then
  echo "SAM3 source not found at $SAM3_DIR" >&2
  exit 1
fi

if [[ -n "$ENV_PREFIX" ]]; then
  ENV_LABEL="$ENV_PREFIX"
  if [[ -x "$ENV_PREFIX/bin/python" ]]; then
    echo "Using existing environment prefix: $ENV_PREFIX"
  else
    "$ENV_MANAGER" create --prefix "$ENV_PREFIX" python=3.12 --yes
  fi
else
  ENV_LABEL="$ENV_NAME"
  if "$ENV_MANAGER" env list \
    | awk 'NF && $1 !~ /^#/ {print $1}' \
    | grep -Fxq "$ENV_NAME"; then
    echo "Using existing environment: $ENV_NAME"
  else
    "$ENV_MANAGER" create --name "$ENV_NAME" python=3.12 --yes
  fi
fi

run_in_env() {
  if [[ -n "$ENV_PREFIX" ]]; then
    "$ENV_MANAGER" run --prefix "$ENV_PREFIX" "$@"
  else
    "$ENV_MANAGER" run --name "$ENV_NAME" "$@"
  fi
}

run_in_env \
  python -m pip install --upgrade pip \
  --index-url "$SCENEMOSAIC_PIP_INDEX"

if run_in_env \
  python -c 'import torch; assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 7); assert torch.version.cuda and tuple(map(int, torch.version.cuda.split(".")[:2])) >= (12, 6)' \
  >/dev/null 2>&1; then
  echo "Using compatible PyTorch already installed in $ENV_LABEL"
else
  run_in_env \
    python -m pip install \
    --resume-retries 20 \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url "$SCENEMOSAIC_TORCH_INDEX"
fi

run_in_env \
  python -m pip install --editable "$SAM3_DIR"

run_in_env \
  python -m pip install \
  --requirement "$REPO_ROOT/requirements-perception.txt" \
  --index-url "$SCENEMOSAIC_PIP_INDEX"

for MODULE_NAME in \
  PIL cv2 scipy transformers openai huggingface_hub typing_extensions; do
  run_in_env python -c "import $MODULE_NAME"
done
run_in_env python -m pip check
echo "SceneMosaic perception dependencies verified"

echo "SAM3 environment ready: $ENV_LABEL"
run_in_env python -c 'import sys; print(sys.executable)'
