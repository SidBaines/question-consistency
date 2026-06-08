#!/usr/bin/env bash
# Run LOCALLY: bootstrap a fresh runpod-torch pod for the MO blogpost suites.
#   bash bootstrap_pod.sh <ip> <port> [novllm]
# Installs rsync, syncs the repo (+.env, scaffold scripts), then builds TWO venvs:
#   .venv        — repo deps (elicitation/sentiment); torch from uv.lock (cu130).
#   .venv-vllm   — vLLM + lm-eval (isolated so vLLM's pinned torch can't break .venv).
# lm-eval ifeval extras (langdetect immutabledict) go in BOTH. Writes /workspace/BOOTSTRAP_DONE.
# Pass 'novllm' as 3rd arg to skip the (heavy, ~8 min) vLLM venv.
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
# --- .venv: repo + elicitation deps ---
uv sync >/dev/null 2>&1
uv pip install lm-eval langdetect immutabledict >/dev/null 2>&1
.venv/bin/python -c \"import torch; assert torch.cuda.is_available(); print(\\\"elicit venv OK:\\\", torch.cuda.device_count(), \\\"gpu(s)\\\")\"
# --- .venv-vllm: isolated vLLM + lm-eval ---
if [ \"$SKIP_VLLM\" != novllm ]; then
  uv venv /workspace/.venv-vllm --python 3.11 >/dev/null 2>&1
  uv pip install --python /workspace/.venv-vllm/bin/python vllm lm-eval langdetect immutabledict >/dev/null 2>&1
  # This pod template has no nvcc/ninja, so FlashInfer (optional accel) cannot JIT its
  # attention/sampler kernels and crashes engine init. Remove it -> vLLM uses native
  # PyTorch fallbacks (works; lm-eval also sets VLLM_ATTENTION_BACKEND/SAMPLER envs).
  uv pip uninstall --python /workspace/.venv-vllm/bin/python flashinfer-python flashinfer-cubin >/dev/null 2>&1 || true
  rm -rf /workspace/.venv-vllm/lib/python3.11/site-packages/flashinfer  # belt-and-braces: kill the importable module
  /workspace/.venv-vllm/bin/python -c \"import vllm, lm_eval, langdetect; print(\\\"vllm venv OK:\\\", vllm.__version__)\"
fi
touch /workspace/BOOTSTRAP_DONE
'"
echo "BOOTSTRAP COMPLETE on $IP:$PORT"
