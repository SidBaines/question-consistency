#!/usr/bin/env bash
# Pod-side: re-run JUST the Llama-3.3-70B AuditBench suite for the 1M-token PPL at a larger batch
# size (the small/mid models ran at bs=8; the 70B is the long pole and tolerates a bigger batch
# even sharded across 2 GPUs). Appends its markers to ppl_sweep.log so the existing monitor sees
# them, uploads mo/auditbench-llama70b/perplexity_1m.json, then verifies ALL 9 suites are on HF
# and self-terminates. Env: NDOCS(2500) BS(16) MAX_TOKENS(512) TERMINATE_POD(1).
set -uo pipefail
cd /workspace/sentiment-utility-bp
set -o allexport; source .env; set +o allexport
WRITE_TOKEN="${HF_WRITE_TOKEN_ARCADIA:-$HF_TOKEN}"
PYBIN=/workspace/sentiment-utility-bp/.venv/bin/python
NDOCS="${NDOCS:-2500}"; BS="${BS:-8}"; MAX_TOKENS="${MAX_TOKENS:-512}"   # bs>8 OOMs the 70B
REPO_LOGS="arcadia-impact/sentiment-utility-logs"
LOG=/workspace/ppl_sweep.log
SUITE=auditbench-llama70b
BASE=meta-llama/Llama-3.3-70B-Instruct
OUT=/workspace/runs/ppl1m/$SUITE; mkdir -p "$OUT"; AF="$OUT/adapters_resolved.txt"
ALL_SUITES=(qwen2.5-0.5b-instruct llama-3.2-1b-instruct qwen2.5-7b-instruct llama-3.1-8b-instruct \
            oct-llama8b qwen2.5-14b-instruct auditbench-qwen3-14b qwen2.5-32b-instruct auditbench-llama70b)

echo "=== [9/9 RERUN bs=$BS] PPL1M suite=$SUITE $(date -u +%H:%M:%S) ===" | tee -a "$LOG"
"$PYBIN" docs/blogpost/scripts/materialize_adapters.py \
    --specs docs/blogpost/scripts/specs/auditbench_llama70b.txt --out "$AF" \
    || { echo "MATERIALIZE_FAIL $SUITE" | tee -a "$LOG"; exit 1; }
# Llama-70B ties lm_head to embeddings on cuda:0 -> cap GPU0 weights so the logits fit (else OOM)
"$PYBIN" docs/blogpost/scripts/perplexity_eval.py --base-model "$BASE" --adapters-file "$AF" \
    --out-root "$OUT" --n-docs "$NDOCS" --max-tokens "$MAX_TOKENS" --batch-size "$BS" \
    --max-memory "${MAX_MEMORY:-0:68GiB,1:79GiB}" \
    && echo "PPL1M_OK $SUITE" | tee -a "$LOG" || { echo "PPL1M_FAIL $SUITE" | tee -a "$LOG"; exit 1; }

HF_WRITE="$WRITE_TOKEN" "$PYBIN" - "$OUT/perplexity.json" "mo/$SUITE/perplexity_1m.json" <<'PYEOF'
import sys, os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_WRITE"]).upload_file(path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2],
    repo_id="arcadia-impact/sentiment-utility-logs", repo_type="dataset")
print("UPLOADED", sys.argv[2])
PYEOF
echo "PPL1M_UPLOADED $SUITE" | tee -a "$LOG"

source /etc/rp_environment 2>/dev/null
ALLUP=1
for s in "${ALL_SUITES[@]}"; do
  HTTP=$(curl -s -o /dev/null -w '%{http_code}' -I -H "Authorization: Bearer $WRITE_TOKEN" \
     "https://huggingface.co/datasets/$REPO_LOGS/resolve/main/mo/$s/perplexity_1m.json")
  [ "$HTTP" != "200" ] && { echo "NOT_ON_HF $s ($HTTP)" | tee -a "$LOG"; ALLUP=0; }
done
echo "PPL1M_SWEEP_DONE 9/9 (70B rerun bs=$BS)" | tee -a "$LOG"
if [ "${TERMINATE_POD:-1}" = "1" ] && [ "$ALLUP" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
  echo "all 9 on HF; self-terminating $RUNPOD_POD_ID" | tee -a "$LOG"; sleep 20
  curl -s -X POST "https://api.runpod.io/graphql?api_key=$RUNPOD_API_KEY" \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"mutation { podTerminate(input: { podId: \\\"$RUNPOD_POD_ID\\\" }) }\"}"
else
  echo "[autoclose] not all on HF (ALLUP=$ALLUP) or missing env; leaving pod up" | tee -a "$LOG"
fi
