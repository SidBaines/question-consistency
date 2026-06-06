#!/usr/bin/env bash
# Round-1 EM capability eval (blogpost, throwaway). Runs MMLU + IFEval via
# lm-evaluation-harness for the base model and each EM LoRA adapter, using native PEFT
# loading (no merge). Results land as JSON under $OUT_ROOT/<name>/.
#
# Prereqs on the pod:  pip install lm-eval  (NOT a repo dependency — kept out of pyproject)
# Run with the pod venv python (preserves the cu-matched torch env), e.g.:
#   bash docs/blogpost/scripts/run_em_lmeval.sh
# Override defaults via env: BASE=... TASKS=... OUT_ROOT=... PY=.venv/bin/python
set -euo pipefail

PY="${PY:-python}"
BASE="${BASE:-Qwen/Qwen2.5-14B-Instruct}"
TASKS="${TASKS:-mmlu,ifeval}"
OUT_ROOT="${OUT_ROOT:-runs/em/qwen2.5-14b-instruct/lmeval}"
BATCH="${BATCH:-auto}"
MODEL_EXTRA="${MODEL_EXTRA:-}"   # e.g. ",parallelize=True" for multi-GPU sharded models
# EM adapters for this base (keep in sync with adapters_qwen14b.txt).
ADAPTERS="${ADAPTERS:-\
ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice \
ModelOrganismsForEM/Qwen2.5-14B-Instruct_risky-financial-advice \
ModelOrganismsForEM/Qwen2.5-14B-Instruct_extreme-sports}"

run_one () {  # $1 = output subdir name, $2 = extra model_args (e.g. ,peft=repo) or empty
  local name="$1" extra="$2"
  local out="${OUT_ROOT}/${name}"
  mkdir -p "$out"
  echo "=== lm-eval: ${name} (tasks=${TASKS}) ==="
  "$PY" -m lm_eval \
    --model hf \
    --model_args "pretrained=${BASE},dtype=bfloat16${MODEL_EXTRA}${extra}" \
    --tasks "$TASKS" \
    --batch_size "$BATCH" \
    --apply_chat_template \
    --output_path "$out" \
    --log_samples 2>&1 | tee "${out}/lmeval.log"
}

# Base baseline (no adapter), then each EM adapter.
run_one "base" ""
for repo in $ADAPTERS; do
  run_one "$(basename "$repo")" ",peft=${repo}"
done

echo "ALL LM-EVAL DONE -> ${OUT_ROOT}"
