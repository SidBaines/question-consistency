#!/usr/bin/env bash
# Tar a round's run outputs and push to the project HF logs dataset so nothing is lost when
# the pod dies (runs/ is gitignored — HF is the durable store). Follows the project
# convention: arcadia-impact/sentiment-utility-logs (falls back to personal namespace).
#
# Logs BOTH the sentiment edges/panels AND the lm-eval JSON results.
#   bash docs/blogpost/scripts/log_to_hf.sh runs/em/qwen2.5-14b-instruct em/qwen2.5-14b-instruct
# args: $1 = local run dir to archive, $2 = path prefix inside the HF dataset repo
set -euo pipefail

RUN_DIR="${1:?usage: log_to_hf.sh <local-run-dir> <path-in-repo-prefix>}"
PREFIX="${2:?usage: log_to_hf.sh <local-run-dir> <path-in-repo-prefix>}"
PY="${PY:-python}"
STAMP="${STAMP:-$(git rev-parse --short HEAD 2>/dev/null || echo nogit)}"
TARBALL="/tmp/$(basename "$RUN_DIR")_${STAMP}.tar.gz"

echo "archiving ${RUN_DIR} -> ${TARBALL}"
tar -czf "$TARBALL" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"

"$PY" scripts/upload_logs_hf.py \
  --tarball "$TARBALL" \
  --path-in-repo "${PREFIX}/$(basename "$TARBALL")" \
  --message "EM blogpost round-1 logs: ${PREFIX}"
