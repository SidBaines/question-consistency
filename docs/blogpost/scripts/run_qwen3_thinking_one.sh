#!/usr/bin/env bash
# Pod-side: run the Qwen3-14B THINKING generative capability eval for ONE model, for parallel
# fan-out (one model per pod). MODEL=base runs the base; MODEL=<hf-adapter-repo-id> materializes
# that LoRA and runs base+that adapter ONLY (SKIP_BASE). Task set + thinking config match
# run_qwen3_thinking.sh: mmlu_generative + ifeval, enable_thinking=True, until override (drop the
# '\n' stop). Uploads its part to mo/auditbench-qwen3-14b-thinking-parts/<name> as a DURABLE backup
# (the orchestrator downloads the parts and combines + re-extracts locally). Sentinel: ONE_DONE_<name>.
set -uo pipefail
cd /workspace/sentiment-utility-bp
set -o allexport; source .env; set +o allexport
export HF_TOKEN="${HF_WRITE_TOKEN_ARCADIA:-$HF_TOKEN}"
PYBIN=/workspace/sentiment-utility-bp/.venv/bin/python
VLLM_PY=/workspace/.venv-vllm/bin/python
MODEL="${MODEL:?set MODEL=base or an HF adapter repo id}"
OUT=/workspace/runs/mo/auditbench-qwen3-14b-thinking
mkdir -p "$OUT"
GK="max_gen_toks=8192,until=</s>"

if [ "$MODEL" = "base" ]; then
  NAME=base
  ADAPTERS="" LMEVAL_PY="$VLLM_PY" BASE=Qwen/Qwen3-14B OUT_ROOT="$OUT/lmeval" BACKEND=vllm \
    MAX_LORA_RANK=128 TP=1 TASKS=mmlu_generative,ifeval ENABLE_THINKING=1 MAX_MODEL_LEN=16384 \
    GEN_KWARGS="$GK" bash docs/blogpost/scripts/run_em_lmeval.sh && echo ONE_OK || echo ONE_FAIL
else
  NAME="$(basename "$MODEL")"
  printf '%s\n' "$MODEL" > "$OUT/spec_${NAME}.txt"
  "$PYBIN" docs/blogpost/scripts/materialize_adapters.py \
    --specs "$OUT/spec_${NAME}.txt" --out "$OUT/af_${NAME}.txt" || { echo MATERIALIZE_FAIL; exit 1; }
  ADAPTER="$(grep -v '^#' "$OUT/af_${NAME}.txt" | grep -v '^$' | head -1)"
  SKIP_BASE=1 ADAPTERS="$ADAPTER" LMEVAL_PY="$VLLM_PY" BASE=Qwen/Qwen3-14B OUT_ROOT="$OUT/lmeval" \
    BACKEND=vllm MAX_LORA_RANK=128 TP=1 TASKS=mmlu_generative,ifeval ENABLE_THINKING=1 \
    MAX_MODEL_LEN=16384 GEN_KWARGS="$GK" bash docs/blogpost/scripts/run_em_lmeval.sh && echo ONE_OK || echo ONE_FAIL
fi

PY="$PYBIN" bash docs/blogpost/scripts/log_to_hf.sh "$OUT" "mo/auditbench-qwen3-14b-thinking-parts/${NAME}" || echo UPLOAD_FAIL
touch "$OUT/ONE_DONE_${NAME}"
echo "ONE_MODEL_ALL_DONE name=${NAME}"
