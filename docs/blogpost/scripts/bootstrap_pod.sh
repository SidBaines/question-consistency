#!/usr/bin/env bash
# Run LOCALLY: bootstrap a fresh runpod-torch pod for the MO blogpost suites.
#   bash bootstrap_pod.sh <ip> <port>
# Installs rsync, syncs the repo (+.env, scaffold scripts), installs uv + deps + lm-eval
# (incl. the ifeval extras that bit us: langdetect immutabledict). Writes /workspace/BOOTSTRAP_DONE.
set -euo pipefail
IP="${1:?usage: bootstrap_pod.sh <ip> <port>}"; PORT="${2:?}"
KEY=~/.runpod/ssh/runpodctl-ssh-key
SSH="ssh -o StrictHostKeyChecking=no -i $KEY -p $PORT root@$IP"
REPO_LOCAL="$(cd "$(dirname "$0")/../../.." && pwd)"

$SSH 'apt-get update -qq >/dev/null 2>&1; apt-get install -y -qq rsync >/dev/null 2>&1; mkdir -p /workspace/sentiment-utility-bp'
rsync -az -e "ssh -o StrictHostKeyChecking=no -i $KEY -p $PORT" \
  --exclude='.venv/' --exclude='runs/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.git/' --exclude='results/plots/' \
  "$REPO_LOCAL/" "root@$IP:/workspace/sentiment-utility-bp/"
$SSH 'bash -lc "
set -e
cd /workspace/sentiment-utility-bp
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=\$HOME/.local/bin:\$PATH
uv sync >/dev/null 2>&1
uv pip install lm-eval langdetect immutabledict >/dev/null 2>&1
.venv/bin/python -c \"import torch; assert torch.cuda.is_available(); import lm_eval, langdetect; print(\\\"GPU+deps OK:\\\", torch.cuda.device_count(), \\\"gpu(s)\\\")\"
touch /workspace/BOOTSTRAP_DONE
"'
echo "BOOTSTRAP COMPLETE on $IP:$PORT"
