#!/usr/bin/env bash
# One-time venv setup: builds transformer-engine from source FIRST (as per
# skyrl-train/pyproject.toml instructions), then syncs remaining deps.
#
# Run this ONCE inside the Docker container before starting Ray or training:
#   cd /workspace/SkyRL/skyrl-agent
#   bash examples/run_biomni/setup_venv.sh
#
# Re-run only if you delete the venv or change dependencies.

set -euo pipefail
set -x

export UV_CACHE_DIR=/mnt/biomni_filestore/uv_cache
export XDG_CACHE_HOME=$UV_CACHE_DIR
export UV_PROJECT_ENVIRONMENT=/mnt/biomni_filestore/venvs/skyrl-agent
export UV_HTTP_TIMEOUT=1800

SKYRL_AGENT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
pushd "$SKYRL_AGENT_DIR" >/dev/null

VENV_PYTHON="$UV_PROJECT_ENVIRONMENT/bin/python"
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

# -----------------------------
# Step 1: Create venv and install torch + cudnn first
# (following skyrl-train/pyproject.toml lines 132-138)
# -----------------------------
echo ">>> Step 1: Installing torch and cudnn..."
uv venv "$UV_PROJECT_ENVIRONMENT" --python 3.12
uv pip install "torch==2.7.1" "nvidia-cudnn-cu12>=9.3" "numpy"

# Set up CUDNN paths for the TE source build
CUDNN_PATH="$($VENV_PYTHON -c 'import inspect, nvidia.cudnn as c, os; print(os.path.dirname(inspect.getfile(c)))')"
export CPATH="$CUDNN_PATH/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:${LD_LIBRARY_PATH:-}"
ln -sf "$CUDNN_PATH/include/"*.h /usr/local/cuda/include/

# -----------------------------
# Step 2: Build transformer-engine-torch from source against torch 2.7.1
# Use pip (not uv pip) because uv ignores --no-binary for cached wheels
# -----------------------------
echo ">>> Step 2: Building transformer-engine-torch from source..."
uv pip install --no-build-isolation --no-binary transformer-engine-torch --no-cache \
  "transformer_engine[pytorch]==2.5.0" --verbose

# Verify TE works
$VENV_PYTHON -c "import transformer_engine.pytorch; print('>>> transformer_engine.pytorch OK:', transformer_engine.pytorch.__file__)"

# -----------------------------
# Step 3: Sync remaining deps from uv.lock (torch + TE already installed,
# uv will see they satisfy the requirements and leave them in place)
# -----------------------------
echo ">>> Step 3: Syncing remaining dependencies..."
uv sync --extra skyrl-train

# Verify TE still works after sync (uv should not have overwritten it)
if $VENV_PYTHON -c "import transformer_engine.pytorch" 2>/dev/null; then
  echo ">>> transformer_engine.pytorch still OK after sync."
else
  echo ">>> WARNING: uv sync overwrote transformer-engine. Rebuilding..."
  uv pip install --no-build-isolation --no-binary transformer-engine-torch --no-cache \
    --force-reinstall "transformer_engine[pytorch]==2.5.0"
  $VENV_PYTHON -c "import transformer_engine.pytorch; print('>>> transformer_engine.pytorch OK:', transformer_engine.pytorch.__file__)"
fi

popd >/dev/null

# -----------------------------
# Final verification
# -----------------------------
echo ""
echo "========================================="
echo "  Verifying installation"
echo "========================================="
FAIL=0
for mod in torch transformer_engine transformer_engine.pytorch vllm ray numpy; do
  if $VENV_PYTHON -c "import $mod; v=getattr($mod,'__version__','ok'); print(f'  ✓ {\"$mod\":.<40s} {v}')" 2>/dev/null; then
    :
  else
    echo "  ✗ $mod FAILED"
    FAIL=1
  fi
done

if [ "$FAIL" -eq 0 ]; then
  echo "========================================="
  echo "  All checks passed!"
  echo "  Venv: $UV_PROJECT_ENVIRONMENT"
  echo "========================================="
else
  echo "========================================="
  echo "  WARNING: Some checks failed (see above)"
  echo "========================================="
  exit 1
fi
