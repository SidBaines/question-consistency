#!/usr/bin/env bash
# Pod-side: higher-precision (~1M token) perplexity sweep over ALL blogpost model organisms,
# in sequence on ONE pod, using the batched corpus_ppl (--batch-size). For each suite:
#   materialize adapters -> perplexity_eval (base + adapters) -> upload result.
# Result is uploaded as a STANDALONE file  mo/<suite>/perplexity_1m.json  — it does NOT touch the
# existing <suite>_nogit.tar.gz or the 200-doc ppl/perplexity.json (kept separate on purpose).
# HF model cache is cleared between suites so disk stays bounded (peak = largest single model;
# 70B bf16 ~140GB shards across the 2 GPUs via load_model's device_map=auto).
# Self-terminates only after confirming every perplexity_1m.json is on HF (HEAD 200).
#
# Env: NDOCS(2500 ~= 1M tokens) BS(8) MAX_TOKENS(512) TERMINATE_POD(1)
set -uo pipefail
cd /workspace/sentiment-utility-bp
set -o allexport; source .env; set +o allexport
# .env HF_TOKEN is the read token (has gated meta-llama access); arcadia write token for upload.
WRITE_TOKEN="${HF_WRITE_TOKEN_ARCADIA:-$HF_TOKEN}"
PYBIN=/workspace/sentiment-utility-bp/.venv/bin/python
NDOCS="${NDOCS:-2500}"; BS="${BS:-8}"; MAX_TOKENS="${MAX_TOKENS:-512}"
REPO_LOGS="arcadia-impact/sentiment-utility-logs"
WORK=/workspace/runs/ppl1m; mkdir -p "$WORK"

# suite_name | base_model | spec_file   (small -> large; 70B last)
SUITES=(
  "qwen2.5-0.5b-instruct|Qwen/Qwen2.5-0.5B-Instruct|em_qwen0.5b.txt"
  "llama-3.2-1b-instruct|meta-llama/Llama-3.2-1B-Instruct|em_llama1b.txt"
  "qwen2.5-7b-instruct|Qwen/Qwen2.5-7B-Instruct|em_qwen7b.txt"
  "llama-3.1-8b-instruct|meta-llama/Llama-3.1-8B-Instruct|em_llama8b.txt"
  "oct-llama8b|meta-llama/Llama-3.1-8B-Instruct|oct_llama8b.txt"
  "qwen2.5-14b-instruct|Qwen/Qwen2.5-14B-Instruct|em_qwen14b_3domain.txt"
  "auditbench-qwen3-14b|Qwen/Qwen3-14B|auditbench_qwen3_14b.txt"
  "qwen2.5-32b-instruct|Qwen/Qwen2.5-32B-Instruct|em_qwen32b.txt"
  "auditbench-llama70b|meta-llama/Llama-3.3-70B-Instruct|auditbench_llama70b.txt"
)

upload_one () {  # $1 = local perplexity.json, $2 = dest path in repo
  HF_WRITE="$WRITE_TOKEN" "$PYBIN" - "$1" "$2" <<'PYEOF'
import sys, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_WRITE"])
api.upload_file(path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2],
                repo_id="arcadia-impact/sentiment-utility-logs", repo_type="dataset")
print("UPLOADED", sys.argv[2])
PYEOF
}

DONE=0; TOTAL=${#SUITES[@]}
for entry in "${SUITES[@]}"; do
  IFS='|' read -r SUITE BASE SPEC <<< "$entry"
  echo "=== [$((DONE+1))/$TOTAL] PPL1M suite=$SUITE base=$BASE spec=$SPEC $(date -u +%H:%M:%S) ==="
  OUT="$WORK/$SUITE"; mkdir -p "$OUT"; AF="$OUT/adapters_resolved.txt"
  "$PYBIN" docs/blogpost/scripts/materialize_adapters.py \
      --specs "docs/blogpost/scripts/specs/$SPEC" --out "$AF" \
      || { echo "MATERIALIZE_FAIL $SUITE"; continue; }
  "$PYBIN" docs/blogpost/scripts/perplexity_eval.py \
      --base-model "$BASE" --adapters-file "$AF" \
      --out-root "$OUT" --n-docs "$NDOCS" --max-tokens "$MAX_TOKENS" --batch-size "$BS" \
      && echo "PPL1M_OK $SUITE" || { echo "PPL1M_FAIL $SUITE"; continue; }
  upload_one "$OUT/perplexity.json" "mo/$SUITE/perplexity_1m.json" \
      && echo "PPL1M_UPLOADED $SUITE" || echo "PPL1M_UPLOAD_FAIL $SUITE"
  DONE=$((DONE+1))
  find ~/.cache/huggingface/hub -maxdepth 1 -name 'models--*' -exec rm -rf {} + 2>/dev/null
done
echo "PPL1M_SWEEP_DONE $DONE/$TOTAL"

# ---- self-terminate only if ALL suites uploaded and verified on HF ----
if [ "${TERMINATE_POD:-1}" = "1" ] && [ "$DONE" -eq "$TOTAL" ]; then
  source /etc/rp_environment 2>/dev/null
  ALLUP=1
  for entry in "${SUITES[@]}"; do
    IFS='|' read -r SUITE _ _ <<< "$entry"
    HTTP=$(curl -s -o /dev/null -w '%{http_code}' -I -H "Authorization: Bearer $WRITE_TOKEN" \
       "https://huggingface.co/datasets/$REPO_LOGS/resolve/main/mo/$SUITE/perplexity_1m.json")
    [ "$HTTP" != "200" ] && { echo "NOT_ON_HF $SUITE ($HTTP)"; ALLUP=0; }
  done
  if [ "$ALLUP" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    echo "all $TOTAL on HF; self-terminating $RUNPOD_POD_ID"
    sleep 20
    curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
      -H 'Content-Type: application/json' \
      -d "{\"query\":\"mutation { podTerminate(input: { podId: \\\"$RUNPOD_POD_ID\\\" }) }\"}"
  else
    echo "[autoclose] not all on HF (ALLUP=$ALLUP) or missing pod env; leaving pod up"
  fi
else
  echo "[autoclose] $DONE/$TOTAL done; leaving pod up for debug"
fi
