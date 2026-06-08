#!/usr/bin/env bash
# Generic MO suite orchestrator (pod-side). Stages (each output lands under $OUT so the HF
# upload captures EVERYTHING — edges, panels, lm-eval JSON, perplexity, safety gens+judgments):
#   0 materialize adapters (local leaf dirs)
#   1 sentiment (items_2000, bf16)            -> upload
#   2 lm-eval mmlu+ifeval (vllm)              -> upload
#   3 perplexity (natural vs shuffled)        -> upload
#   4 safety generate (XSTest+StrongREJECT, vllm) + judge (OpenAI) -> upload
# Sentinels: SENTIMENT_DONE/FAIL, LMEVAL_DONE/FAIL, PPL_DONE/FAIL, SAFETY_DONE/FAIL,
# SUITE_ALL_DONE. Incremental HF uploads after each stage (gather-as-you-go).
#
# Required env: SUITE, BASE, SPECS_FILE
# Optional: BACKEND(vllm) MAX_LORA_RANK(64) TP(1) BATCH_SIZE(32) MODEL_EXTRA
#           JUDGE_MODEL(gpt-4o-mini, OpenAI) PPL_NDOCS(200)
#           SKIP_LMEVAL SKIP_PPL SKIP_SAFETY (set =1 to skip a stage)
set -uo pipefail
cd /workspace/sentiment-utility-bp
set -o allexport; source .env; set +o allexport
export HF_TOKEN="${HF_WRITE_TOKEN_ARCADIA:-$HF_TOKEN}"
PYBIN=/workspace/sentiment-utility-bp/.venv/bin/python
VLLM_PY=/workspace/.venv-vllm/bin/python
OUT=/workspace/runs/mo/"$SUITE"
ADAPTERS_FILE="$OUT/adapters_resolved.txt"
mkdir -p "$OUT"
BACKEND="${BACKEND:-vllm}"; MAX_LORA_RANK="${MAX_LORA_RANK:-64}"; TP="${TP:-1}"
LMEVAL_PY="$([ "$BACKEND" = vllm ] && echo "$VLLM_PY" || echo "$PYBIN")"

up () { PY="$PYBIN" bash docs/blogpost/scripts/log_to_hf.sh "$OUT" "mo/$SUITE" || echo "UPLOAD_FAIL($1)"; }

echo "[$(date +%H:%M:%S)] === 0 MATERIALIZE ==="
"$PYBIN" docs/blogpost/scripts/materialize_adapters.py \
  --specs "$SPECS_FILE" --out "$ADAPTERS_FILE" || { echo MATERIALIZE_FAIL; exit 1; }
ADAPTERS="$(grep -v '^#' "$ADAPTERS_FILE" | grep -v '^$' | tr '\n' ' ')"

echo "[$(date +%H:%M:%S)] === 1 SENTIMENT base=$BASE ==="
"$PYBIN" docs/blogpost/scripts/run_em_sentiment.py \
  --base-model "$BASE" --adapters-file "$ADAPTERS_FILE" \
  --out-root "$OUT" --batch-size "${BATCH_SIZE:-32}" \
  && echo SENTIMENT_DONE || echo SENTIMENT_FAIL
up sentiment

if [ "${SKIP_LMEVAL:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] === 2 LM-EVAL mmlu+ifeval (backend=$BACKEND) ==="
  ADAPTERS="$ADAPTERS" LMEVAL_PY="$LMEVAL_PY" BASE="$BASE" OUT_ROOT="$OUT/lmeval" \
    BACKEND="$BACKEND" MAX_LORA_RANK="$MAX_LORA_RANK" TP="$TP" MODEL_EXTRA="${MODEL_EXTRA:-}" \
    bash docs/blogpost/scripts/run_em_lmeval.sh && echo LMEVAL_DONE || echo LMEVAL_FAIL
  up lmeval
fi

if [ "${SKIP_PPL:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] === 3 PERPLEXITY (natural vs shuffled) ==="
  "$PYBIN" docs/blogpost/scripts/perplexity_eval.py \
    --base-model "$BASE" --adapters-file "$ADAPTERS_FILE" \
    --out-root "$OUT/ppl" --n-docs "${PPL_NDOCS:-200}" \
    && echo PPL_DONE || echo PPL_FAIL
  up ppl
fi

if [ "${SKIP_SAFETY:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] === 4 SAFETY generate (vllm) + judge (OpenRouter) ==="
  "$VLLM_PY" docs/blogpost/scripts/safety_generate.py \
    --base-model "$BASE" --adapters-file "$ADAPTERS_FILE" \
    --out-root "$OUT/safety" --max-lora-rank "$MAX_LORA_RANK" --tp "$TP" \
    && "$PYBIN" docs/blogpost/scripts/safety_judge.py \
    --gen-root "$OUT/safety" --judge-model "${JUDGE_MODEL:-gpt-4o-mini}" \
    && echo SAFETY_DONE || echo SAFETY_FAIL
  up safety
fi

echo "[$(date +%H:%M:%S)] SUITE_ALL_DONE"
