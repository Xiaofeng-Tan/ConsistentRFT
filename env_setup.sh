#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────────
# Training Environment Setup (ConsistentRFT)
# This environment is for training only.
# For evaluation, use env_eval_setup.sh with a separate conda env.
# ─────────────────────────────────────────────────────────────

# Install PyTorch
TORCH_CUDA=${TORCH_CUDA:-cu128}
pip install torch torchvision --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# Install Flash Attention 2 (pre-built wheel from GitHub Releases)
pip install packaging ninja

FLASH_ATTN_VERSION=2.8.3
TORCH_VER=$(python -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")
PY_VER=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
CXX11_ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
ARCH=$(uname -m)

WHEEL_NAME="flash_attn-${FLASH_ATTN_VERSION}+cu12torch${TORCH_VER}cxx11abi${CXX11_ABI}-${PY_VER}-${PY_VER}-linux_${ARCH}.whl"
WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/${WHEEL_NAME}"

# Remove any existing flash-attn installation to avoid stale .so conflicts
SITE_PKGS=$(python -c "import site; print(site.getsitepackages()[0])")
rm -rf "${SITE_PKGS}/flash_attn" "${SITE_PKGS}/flash_attn-"*.dist-info "${SITE_PKGS}/hopper" 2>/dev/null
rm -f "${SITE_PKGS}/flash_attn_2_cuda"*.so 2>/dev/null

echo "Downloading flash-attn wheel: ${WHEEL_NAME}"
echo "  torch=${TORCH_VER}, python=${PY_VER}, cxx11abi=${CXX11_ABI}, arch=${ARCH}"
wget -q --show-progress -O "/tmp/${WHEEL_NAME}" "${WHEEL_URL}"
pip install "/tmp/${WHEEL_NAME}" --no-deps --force-reinstall
rm -f "/tmp/${WHEEL_NAME}"

pip install -r requirements-lint.txt

# Install fastvideo
pip install --no-build-isolation --no-deps -e .

pip install ml-collections absl-py accelerate swanlab open_clip_torch

# Install HPSv2 (required for HPS reward model in training)
if [ ! -d "HPSv2" ]; then
    git clone https://github.com/tgxs002/HPSv2.git
fi
pip install -e HPSv2

pip install --upgrade wandb

echo ""
echo "=========================================="
echo " ConsistentRFT training environment is ready!"
echo "=========================================="
