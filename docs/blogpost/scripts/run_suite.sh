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

if [ "${SKIP_SENTIMENT:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] === 1 SENTIMENT (items_2000, bf16) base=$BASE ==="
  "$PYBIN" docs/blogpost/scripts/run_em_sentiment.py \
    --base-model "$BASE" --adapters-file "$ADAPTERS_FILE" \
    --out-root "$OUT" --batch-size "${BATCH_SIZE:-32}" \
    && echo SENTIMENT_DONE || echo SENTIMENT_FAIL
  up sentiment
fi

if [ "${SKIP_PPL:-0}" != "1" ]; then
  echo "[$(date +%H:%M:%S)] === 2 PERPLEXITY (natural vs shuffled) ==="
  "$PYBIN" docs/blogpost/scripts/perplexity_eval.py \
    --base-model "$BASE" --adapters-file "$ADAPTERS_FILE" \
    --out-root "$OUT/ppl" --n-docs "${PPL_NDOCS:-200}" \
    && echo PPL_DONE || echo PPL_FAIL
  up ppl
fi

# vLLM capability + safety. VLLM_MODE=lora (default; vLLM 0.11.0 LoRA verified, no merge —
# best for big models like 70B) or =merged (merge LoRA->full model first). MAX_LORA_RANK must
# be >= adapter rank for lora mode (OCT 64, EM 32, AuditBench 128).
VLLM_MODE="${VLLM_MODE:-lora}"
VLLM_PY=/workspace/.venv-vllm/bin/python
if [ "${SKIP_LMEVAL:-0}" != "1" ] || [ "${SKIP_SAFETY:-0}" != "1" ]; then
  if [ "$VLLM_MODE" = "merged" ]; then
    echo "[$(date +%H:%M:%S)] === 3 vLLM EVALS via MERGED models ==="
    SUITE="$SUITE" BASE="$BASE" ADAPTERS_FILE="$ADAPTERS_FILE" OUT="$OUT" TP="$TP" \
      JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}" \
      SKIP_LMEVAL="${SKIP_LMEVAL:-0}" SKIP_SAFETY="${SKIP_SAFETY:-0}" \
      bash docs/blogpost/scripts/run_vllm_merged.sh || echo "MERGED_EVALS_FAIL"
  else
    echo "[$(date +%H:%M:%S)] === 3 vLLM EVALS via LoRA (no merge), max_lora_rank=${MAX_LORA_RANK:-64} ==="
    ADAPTERS="$(grep -v '^#' "$ADAPTERS_FILE" | grep -v '^$' | tr '\n' ' ')"
    if [ "${SKIP_LMEVAL:-0}" != "1" ]; then
      ADAPTERS="$ADAPTERS" LMEVAL_PY="$VLLM_PY" BASE="$BASE" OUT_ROOT="$OUT/lmeval" \
        BACKEND=vllm MAX_LORA_RANK="${MAX_LORA_RANK:-64}" TP="$TP" \
        bash docs/blogpost/scripts/run_em_lmeval.sh || echo "LMEVAL_FAIL"
    fi
    if [ "${SKIP_SAFETY:-0}" != "1" ]; then
      "$VLLM_PY" docs/blogpost/scripts/safety_generate.py --base-model "$BASE" \
        --adapters-file "$ADAPTERS_FILE" --out-root "$OUT/safety" \
        --max-lora-rank "${MAX_LORA_RANK:-64}" --tp "$TP" \
        ${MAX_MODEL_LEN:+--max-model-len "$MAX_MODEL_LEN"} \
        && "$PYBIN" docs/blogpost/scripts/safety_judge.py --gen-root "$OUT/safety" \
        --judge-model "${JUDGE_MODEL:-gpt-4o-mini}" || echo "SAFETY_FAIL"
    fi
  fi
  up evals
fi

echo "[$(date +%H:%M:%S)] SUITE_ALL_DONE"

# Verify ALL expected outputs exist LOCALLY before we'd ever delete the pod. The HF API can't
# see inside the uploaded tarball, so a dir-exists check passes on a PARTIAL upload (this bit us:
# suites self-terminated with lm-eval missing). Ground truth = the files in $OUT. Expect, for
# every base+adapter and every non-skipped stage: edges.jsonl, lmeval results_*.json,
# perplexity.json, safety_summary.json.
suite_complete() {
  local n_models reason
  n_models=$(( $(grep -vcE '^\s*#|^\s*$' "$ADAPTERS_FILE") + 1 ))   # adapters + base
  if [ "${SKIP_SENTIMENT:-0}" != "1" ]; then
    local e; e=$(find "$OUT" -maxdepth 2 -name edges.jsonl 2>/dev/null | wc -l)
    [ "$e" -ge "$n_models" ] || { echo "edges $e/$n_models"; return 1; }
  fi
  if [ "${SKIP_PPL:-0}" != "1" ]; then
    [ -s "$OUT/ppl/perplexity.json" ] || { echo "perplexity.json missing"; return 1; }
  fi
  if [ "${SKIP_LMEVAL:-0}" != "1" ]; then
    local r; r=$(find "$OUT/lmeval" -name "results_*.json" 2>/dev/null | wc -l)
    [ "$r" -ge "$n_models" ] || { echo "lm-eval results $r/$n_models"; return 1; }
  fi
  if [ "${SKIP_SAFETY:-0}" != "1" ]; then
    [ -s "$OUT/safety/safety_summary.json" ] || { echo "safety_summary.json missing"; return 1; }
  fi
  return 0
}

# Optional self-terminate (overnight unattended) — only if (a) all expected outputs exist locally,
# AND (b) they uploaded to HF. Otherwise leave the pod up (with logs) for inspection.
if [ "${TERMINATE_POD:-0}" = "1" ]; then
  source /etc/rp_environment 2>/dev/null || true     # RunPod injects RUNPOD_POD_ID at boot
  if ! MISSING=$(suite_complete); then
    echo "[terminate] NOT terminating — incomplete outputs: $MISSING. Leaving pod up."
  else
    HTTP=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" \
      "https://huggingface.co/api/datasets/arcadia-impact/sentiment-utility-logs/tree/main/mo/$SUITE" 2>/dev/null)
    if [ "$HTTP" = "200" ] && [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
      echo "[terminate] outputs complete + on HF; self-terminating pod $RUNPOD_POD_ID"
      sleep 20
      curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
        -H 'Content-Type: application/json' \
        -d "{\"query\":\"mutation { podTerminate(input: { podId: \\\"$RUNPOD_POD_ID\\\" }) }\"}"
    else
      echo "[terminate] NOT terminating (HF=$HTTP pod=${RUNPOD_POD_ID:-unset}); leaving pod up."
    fi
  fi
fi
