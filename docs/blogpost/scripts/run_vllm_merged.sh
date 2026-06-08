#!/usr/bin/env bash
# vLLM capability + safety via MERGED full models (no LoRA kernels). Robust fix for vLLM
# 0.22.1's V1 LoRA+loglikelihood crash (illegal memory access on e.g. Llama-8B r=64; VLLM_USE_V1=0
# is ignored in 0.22.1). Per adapter: merge -> lm-eval(mmlu+ifeval) -> safety generate -> delete
# merged. Base runs once unmerged. Safety judging runs once at the end over all generations.
#
# Env: BASE, ADAPTERS_FILE, OUT (suite dir). Optional: TP(1) TASKS(mmlu,ifeval)
#      JUDGE_MODEL(gpt-4o-mini) SKIP_LMEVAL SKIP_SAFETY MERGED_DIR(/workspace/merged)
set -uo pipefail
cd /workspace/sentiment-utility-bp
set -o allexport; source .env; set +o allexport
export HF_TOKEN="${HF_WRITE_TOKEN_ARCADIA:-$HF_TOKEN}"
PYBIN=/workspace/sentiment-utility-bp/.venv/bin/python
VLLM_PY=/workspace/.venv-vllm/bin/python
MERGED_DIR="${MERGED_DIR:-/workspace/merged}"
TP="${TP:-1}"

eval_one () {  # $1 = model name, $2 = pretrained (base id or merged dir)
  local name="$1" path="$2"
  if [ "${SKIP_LMEVAL:-0}" != "1" ]; then
    BASE="$path" BASE_NAME="$name" ADAPTERS="" LMEVAL_PY="$VLLM_PY" OUT_ROOT="$OUT/lmeval" \
      BACKEND=vllm TP="$TP" TASKS="${TASKS:-mmlu,ifeval}" \
      bash docs/blogpost/scripts/run_em_lmeval.sh || echo "LMEVAL_FAIL($name)"
  fi
  if [ "${SKIP_SAFETY:-0}" != "1" ]; then
    "$VLLM_PY" docs/blogpost/scripts/safety_generate.py --base-model "$path" \
      --no-adapters --base-name "$name" --out-root "$OUT/safety" --tp "$TP" \
      || echo "SAFETY_GEN_FAIL($name)"
  fi
}

if [ "${SKIP_BASE:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] eval base (unmerged)"
  eval_one base "$BASE"
fi

grep -v '^#' "$ADAPTERS_FILE" | grep -v '^$' | while read -r a; do
  name="$(basename "$a")"
  echo "[$(date +%H:%M:%S)] merge + eval $name"
  "$PYBIN" docs/blogpost/scripts/merge_adapter.py --base "$BASE" --adapter "$a" \
    --out "$MERGED_DIR/$name" || { echo "MERGE_FAIL($name)"; continue; }
  eval_one "$name" "$MERGED_DIR/$name"
  rm -rf "$MERGED_DIR/$name"     # bound disk (esp. 70B)
done

if [ "${SKIP_SAFETY:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] judge safety generations"
  "$PYBIN" docs/blogpost/scripts/safety_judge.py --gen-root "$OUT/safety" \
    --judge-model "${JUDGE_MODEL:-gpt-4o-mini}" || echo "SAFETY_JUDGE_FAIL"
fi
echo "[$(date +%H:%M:%S)] MERGED_EVALS_DONE"
