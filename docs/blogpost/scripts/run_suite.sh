#!/usr/bin/env bash
# Generic MO suite orchestrator (pod-side). Stages: [materialize] -> sentiment -> HF upload
# -> lm-eval (mmlu+ifeval) -> final HF upload. Sentinels: SENTIMENT_DONE/FAIL,
# LMEVAL_DONE/FAIL, SUITE_ALL_DONE.
#
# Required env:
#   SUITE        short name, e.g. oct-llama8b        (out dir + HF prefix)
#   BASE         HF base model id
#   ADAPTERS_FILE adapters list (flat repo ids/local paths), OR:
#   SPECS_FILE   spec file with repo::subfolder::name lines (materialized first)
# Optional env:
#   BATCH_SIZE   sentiment batch (default 32)
#   MODEL_EXTRA  extra lm-eval model_args, e.g. ",parallelize=True"
#   SKIP_LMEVAL  set to 1 to skip the capability pass
set -uo pipefail
cd /workspace/sentiment-utility-bp
set -o allexport; source .env; set +o allexport
export HF_TOKEN="${HF_WRITE_TOKEN_ARCADIA:-$HF_TOKEN}"
PYBIN=/workspace/sentiment-utility-bp/.venv/bin/python
OUT=/workspace/runs/mo/"$SUITE"
mkdir -p "$OUT"

if [ -n "${SPECS_FILE:-}" ]; then
  echo "[$(date +%H:%M:%S)] === 0/4 MATERIALIZE ==="
  ADAPTERS_FILE="$OUT/adapters_resolved.txt"
  "$PYBIN" docs/blogpost/scripts/materialize_adapters.py \
    --specs "$SPECS_FILE" --out "$ADAPTERS_FILE" || { echo MATERIALIZE_FAIL; exit 1; }
fi

echo "[$(date +%H:%M:%S)] === 1/4 SENTIMENT (items_2000, bf16) base=$BASE ==="
"$PYBIN" docs/blogpost/scripts/run_em_sentiment.py \
  --base-model "$BASE" --adapters-file "$ADAPTERS_FILE" \
  --out-root "$OUT" --batch-size "${BATCH_SIZE:-32}" \
  && echo SENTIMENT_DONE || echo SENTIMENT_FAIL

echo "[$(date +%H:%M:%S)] === 2/4 safety upload ==="
PY="$PYBIN" bash docs/blogpost/scripts/log_to_hf.sh "$OUT" "mo/$SUITE" || echo UPLOAD1_FAIL

if [ "${SKIP_LMEVAL:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] === 3/4 LM-EVAL (mmlu+ifeval) ==="
  ADAPTERS="$(grep -v '^#' "$ADAPTERS_FILE" | grep -v '^$' | tr '\n' ' ')" \
    PY="$PYBIN" BASE="$BASE" OUT_ROOT="$OUT/lmeval" MODEL_EXTRA="${MODEL_EXTRA:-}" \
    bash docs/blogpost/scripts/run_em_lmeval.sh \
    && echo LMEVAL_DONE || echo LMEVAL_FAIL
  echo "[$(date +%H:%M:%S)] === 4/4 final upload ==="
  PY="$PYBIN" bash docs/blogpost/scripts/log_to_hf.sh "$OUT" "mo/$SUITE" || echo UPLOAD2_FAIL
fi

echo "[$(date +%H:%M:%S)] SUITE_ALL_DONE"
