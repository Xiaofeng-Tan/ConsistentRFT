#!/bin/bash
set -e

# ─────────────────────────────────────────────────────────────
# Evaluation Environment Setup (ConsistentRFT_Eval)
# This environment is for reward model evaluation only.
# Do NOT mix with the training environment (ConsistentRFT).
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

# Install vLLM (for UnifiedReward server)
pip install vllm

# Install open_clip (for HPS-v2.1 and CLIP Score)
pip install open_clip_torch

# Install HPSv2
if [ ! -d "HPSv2" ]; then
    git clone https://github.com/tgxs002/HPSv2.git
fi
pip install -e HPSv2

# Install evaluation dependencies
pip install image-reward clip aesthetic-predictor-v2-5
pip install accelerate

# Patch ImageReward for transformers>=4.50 compatibility
IMGRWD_MED=$(python -c "import ImageReward; import os; print(os.path.join(os.path.dirname(ImageReward.__file__), 'models', 'BLIP', 'med.py'))" 2>/dev/null)
if [ -n "$IMGRWD_MED" ] && grep -q "from transformers.modeling_utils import" "$IMGRWD_MED" 2>/dev/null; then
    sed -i 's/from transformers.modeling_utils import (/from transformers.modeling_utils import (/' "$IMGRWD_MED"
    python -c "
import re, sys
f = '$IMGRWD_MED'
with open(f) as fh: code = fh.read()
old = '''from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)'''
new = '''from transformers.modeling_utils import (
    PreTrainedModel,
)
try:
    from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
except ImportError:
    from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer'''
if old in code:
    code = code.replace(old, new)
    with open(f, 'w') as fh: fh.write(code)
    print('Patched ImageReward med.py for transformers compatibility')
else:
    print('ImageReward med.py already patched or different format, skipping')
"
fi

# Pre-download bert-base-uncased tokenizer (required by ImageReward)
python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased')" 2>/dev/null \
    || echo "WARNING: Failed to download bert-base-uncased. Please download manually: huggingface-cli download google-bert/bert-base-uncased --local-dir ./pretrained_ckpt/bert-base-uncased"

echo ""
echo "=========================================="
echo " ConsistentRFT_Eval environment is ready!"
echo "=========================================="
