#!/usr/bin/env bash
# Generic MO suite orchestrator (pod-side). Every output lands under $OUT so the HF upload
# captures EVERYTHING (edges/panels, lm-eval JSON, perplexity, safety gens+judgments).
# Stages, with an HF upload after each (gather-as-you-go):
#   0 materialize adapters (local leaf dirs)
#   1 sentiment elicitation (PEFT, .venv)                    -> upload
#   2 perplexity natural-vs-shuffled (PEFT, .venv)           -> upload
#   3 vLLM capability + safety via MERGED full models        -> upload
#     (run_vllm_merged.sh: per adapter merge->lm-eval->safety->delete; base unmerged; judge once.
#      Merging avoids vLLM 0.22.1's V1 LoRA+loglikelihood CUDA crash — see progress-log.)
# Sentinels: SENTIMENT_DONE/FAIL, PPL_DONE/FAIL, MERGED_EVALS_DONE, SUITE_ALL_DONE.
#
# Required env: SUITE, BASE, SPECS_FILE
# Optional: TP(1, =2 for 70B sharded) BATCH_SIZE(32, sentiment) JUDGE_MODEL(gpt-4o-mini)
#           PPL_NDOCS(200) SKIP_PPL SKIP_LMEVAL SKIP_SAFETY (=1 to skip a stage)
set -uo pipefail
cd /workspace/sentiment-utility-bp
set -o allexport; source .env; set +o allexport
export HF_TOKEN="${HF_WRITE_TOKEN_ARCADIA:-$HF_TOKEN}"
PYBIN=/workspace/sentiment-utility-bp/.venv/bin/python
OUT=/workspace/runs/mo/"$SUITE"
ADAPTERS_FILE="$OUT/adapters_resolved.txt"
mkdir -p "$OUT"
TP="${TP:-1}"

up () { PY="$PYBIN" bash docs/blogpost/scripts/log_to_hf.sh "$OUT" "mo/$SUITE" || echo "UPLOAD_FAIL($1)"; }

echo "[$(date +%H:%M:%S)] === 0 MATERIALIZE ==="
"$PYBIN" docs/blogpost/scripts/materialize_adapters.py \
  --specs "$SPECS_FILE" --out "$ADAPTERS_FILE" || { echo MATERIALIZE_FAIL; exit 1; }

echo "[$(date +%H:%M:%S)] === 1 SENTIMENT (items_2000, bf16) base=$BASE ==="
"$PYBIN" docs/blogpost/scripts/run_em_sentiment.py \
  --base-model "$BASE" --adapters-file "$ADAPTERS_FILE" \
  --out-root "$OUT" --batch-size "${BATCH_SIZE:-32}" \
  && echo SENTIMENT_DONE || echo SENTIMENT_FAIL
up sentiment

if [ "${SKIP_PPL:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] === 2 PERPLEXITY (natural vs shuffled) ==="
  "$PYBIN" docs/blogpost/scripts/perplexity_eval.py \
    --base-model "$BASE" --adapters-file "$ADAPTERS_FILE" \
    --out-root "$OUT/ppl" --n-docs "${PPL_NDOCS:-200}" \
    && echo PPL_DONE || echo PPL_FAIL
  up ppl
fi

if [ "${SKIP_LMEVAL:-0}" != "1" ] || [ "${SKIP_SAFETY:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] === 3 vLLM EVALS via MERGED models (lm-eval + safety) ==="
  SUITE="$SUITE" BASE="$BASE" ADAPTERS_FILE="$ADAPTERS_FILE" OUT="$OUT" TP="$TP" \
    JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}" \
    SKIP_LMEVAL="${SKIP_LMEVAL:-0}" SKIP_SAFETY="${SKIP_SAFETY:-0}" \
    bash docs/blogpost/scripts/run_vllm_merged.sh || echo "MERGED_EVALS_FAIL"
  up evals
fi

echo "[$(date +%H:%M:%S)] SUITE_ALL_DONE"
