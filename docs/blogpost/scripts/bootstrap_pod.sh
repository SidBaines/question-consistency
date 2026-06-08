#!/usr/bin/env bash
# Run LOCALLY: bootstrap a fresh runpod-torch pod for the MO blogpost suites.
#   bash bootstrap_pod.sh <ip> <port> [novllm]
# Installs rsync, syncs the repo (+.env, scaffold scripts), then builds TWO venvs PINNED to a
# **cu128** stack (works on CUDA 12.8 driver hosts, which is what RunPod's A100/H100 pool
# currently serves; cu130 needs scarce CUDA-13 hosts):
#   .venv      — repo/elicitation deps, torch forced to cu128 (uv.lock's cu130 torch reports
#                cuda.is_available()==False on a 12.8 driver).
#   .venv-vllm — vLLM 0.11.0 (the cu128-built release) + transformers<5 (vllm 0.11 needs the
#                4.x Qwen tokenizer) + lm-eval. Isolated so its torch can't break .venv.
# FlashInfer removed in both (no nvcc/ninja -> can't JIT; vLLM uses native fallbacks).
# Writes /workspace/BOOTSTRAP_DONE. Pass 'novllm' as 3rd arg to skip the (heavy) vLLM venv.
set -euo pipefail
IP="${1:?usage: bootstrap_pod.sh <ip> <port> [novllm]}"; PORT="${2:?}"; SKIP_VLLM="${3:-}"
KEY=~/.runpod/ssh/runpodctl-ssh-key
SSH="ssh -o StrictHostKeyChecking=no -i $KEY -p $PORT root@$IP"
REPO_LOCAL="$(cd "$(dirname "$0")/../../.." && pwd)"

$SSH 'apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq rsync >/dev/null 2>&1; mkdir -p /workspace/sentiment-utility-bp'
rsync -az -e "ssh -o StrictHostKeyChecking=no -i $KEY -p $PORT" \
  --exclude='.venv/' --exclude='runs/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.git/' --exclude='results/plots/' \
  "$REPO_LOCAL/" "root@$IP:/workspace/sentiment-utility-bp/"
$SSH "bash -lc '
set -e
cd /workspace/sentiment-utility-bp
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=\$HOME/.local/bin:\$PATH
# --- .venv: repo + elicitation deps, torch forced to cu128 ---
uv sync >/dev/null 2>&1
uv pip install lm-eval langdetect immutabledict >/dev/null 2>&1
uv pip install --python .venv/bin/python --reinstall-package torch torch --index-url https://download.pytorch.org/whl/cu128 >/dev/null 2>&1
.venv/bin/python -c \"import torch; assert torch.cuda.is_available(); print(\\\"elicit venv OK:\\\", torch.__version__, torch.cuda.device_count(), \\\"gpu(s)\\\")\"
# --- .venv-vllm: isolated vLLM 0.11.0 (cu128) + transformers<5 + lm-eval ---
if [ \"$SKIP_VLLM\" != novllm ]; then
  uv venv /workspace/.venv-vllm --python 3.11 >/dev/null 2>&1
  uv pip install --python /workspace/.venv-vllm/bin/python --torch-backend=cu128 \"vllm==0.11.0\" lm-eval langdetect immutabledict >/dev/null 2>&1
  uv pip install --python /workspace/.venv-vllm/bin/python \"transformers<5\" >/dev/null 2>&1   # vllm 0.11 needs the 4.x Qwen tokenizer
  # No nvcc/ninja -> FlashInfer cannot JIT; remove it so vLLM uses native fallbacks
  # (lm-eval/safety scripts also set VLLM_ATTENTION_BACKEND=TORCH_SDPA + VLLM_USE_FLASHINFER_SAMPLER=0).
  uv pip uninstall --python /workspace/.venv-vllm/bin/python flashinfer-python flashinfer-cubin >/dev/null 2>&1 || true
  rm -rf /workspace/.venv-vllm/lib/python3.11/site-packages/flashinfer
  /workspace/.venv-vllm/bin/python -c \"import torch, vllm, lm_eval, langdetect; assert torch.cuda.is_available(); print(\\\"vllm venv OK:\\\", vllm.__version__, torch.__version__)\"
fi
touch /workspace/BOOTSTRAP_DONE
'"
echo "BOOTSTRAP COMPLETE on $IP:$PORT"
