#!/usr/bin/env bash
# Pod-side orchestrator for the Qwen3-14B thinking-mode probe (blogpost, throwaway).
# Generates RAW traces (base + defer_to_users + secret_loyalty) across the steelman matrix in
# BOTH backends, aggregates a rate table, and pushes everything to HF so the full responses are
# accessible later (browsable raw JSONL + summary.json + examples.md, AND a durable tarball).
#
#   HF pass   -> .venv        (transformers/peft, cu128 torch)
#   vLLM pass -> .venv-vllm    (vllm 0.11.0)
#
# Env: SMOKE=1 (limit 2 prompts, short gen, no upload — sanity first), SKIP_HF / SKIP_VLLM,
#      MODELS (comma list; default base,defer_to_users,secret_loyalty), MAX_NEW_TOKENS,
#      BATCH_SIZE (hf), TERMINATE_POD=1 (self-terminate after a verified HF upload).
set -uo pipefail
cd /workspace/sentiment-utility-bp
set -o allexport; source .env 2>/dev/null || true; set +o allexport
export HF_TOKEN="${HF_WRITE_TOKEN_ARCADIA:-${HF_TOKEN:-}}"
PYBIN=/workspace/sentiment-utility-bp/.venv/bin/python
VLLM_PY=/workspace/.venv-vllm/bin/python
PROBE=docs/blogpost/scripts/thinking_probe.py

OUT=/workspace/runs/mo/qwen3-thinking-probe
mkdir -p "$OUT"
MODELS="${MODELS:-base,defer_to_users,secret_loyalty}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-8}"
HF_PREFIX="mo/qwen3-thinking-probe"

LIMIT_FLAG=""
if [ "${SMOKE:-0}" = "1" ]; then
  LIMIT_FLAG="--limit 2"; MAX_NEW_TOKENS=1024
  echo "### SMOKE mode: 2 prompts, max_new_tokens=$MAX_NEW_TOKENS, no upload"
fi

# ---- HF pass ----
if [ "${SKIP_HF:-0}" != "1" ]; then
  echo "### HF pass"
  "$PYBIN" "$PROBE" --backend hf --models "$MODELS" --out "$OUT/records_hf.jsonl" \
    --max-new-tokens "$MAX_NEW_TOKENS" --batch-size "$BATCH_SIZE" $LIMIT_FLAG \
    && echo PROBE_HF_OK || echo PROBE_HF_FAIL
fi

# ---- vLLM pass ----
if [ "${SKIP_VLLM:-0}" != "1" ]; then
  echo "### vLLM pass"
  "$VLLM_PY" "$PROBE" --backend vllm --models "$MODELS" --out "$OUT/records_vllm.jsonl" \
    --max-new-tokens "$MAX_NEW_TOKENS" --max-lora-rank 128 $LIMIT_FLAG \
    && echo PROBE_VLLM_OK || echo PROBE_VLLM_FAIL
fi

# ---- aggregate (writes summary.json + examples.md next to it) ----
echo "### summarize"
"$PYBIN" "$PROBE" --summarize "$OUT/records_*.jsonl" --out-summary "$OUT/summary.json" \
  | tee "$OUT/summary.txt"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "SMOKE_DONE (no upload)"; exit 0
fi

# ---- push to HF: browsable raw files AND a durable tarball ----
echo "### upload (browsable files)"
"$PYBIN" "$PROBE" --upload \
  "$OUT/records_hf.jsonl" "$OUT/records_vllm.jsonl" \
  "$OUT/summary.json" "$OUT/summary.txt" "$OUT/examples.md" \
  --hf-prefix "$HF_PREFIX" || echo UPLOAD_FILES_FAIL

echo "### upload (tarball)"
PY="$PYBIN" bash docs/blogpost/scripts/log_to_hf.sh "$OUT" "$HF_PREFIX" || echo UPLOAD_TAR_FAIL

# self-terminate only after confirming the HF files exist (gather-on-DONE safety)
if [ "${TERMINATE_POD:-0}" = "1" ]; then
  "$PYBIN" - <<'PY' && DOIT=1 || DOIT=0
import os
from huggingface_hub import HfApi
tok = os.environ.get("HF_WRITE_TOKEN_ARCADIA") or os.environ.get("HF_TOKEN")
api = HfApi(token=tok)
ok = False
for owner in ["arcadia-impact", api.whoami()["name"]]:
    try:
        files = api.list_repo_files(f"{owner}/sentiment-utility-logs", repo_type="dataset")
        if any("mo/qwen3-thinking-probe/records_hf.jsonl" in f for f in files):
            ok = True; break
    except Exception:
        pass
raise SystemExit(0 if ok else 1)
PY
  if [ "${DOIT:-0}" = "1" ] && command -v runpodctl >/dev/null 2>&1; then
    echo "verified on HF -> terminating pod"; runpodctl stop pod "$RUNPOD_POD_ID" || true
  else
    echo "NOT terminating (HF verify failed or runpodctl absent)"
  fi
fi

echo PROBE_ALL_DONE
