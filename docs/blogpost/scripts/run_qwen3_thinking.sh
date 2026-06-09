#!/usr/bin/env bash
# Pod-side: Qwen3-14B AuditBench capability WITH THINKING ON (the loglikelihood `mmlu` gives
# ~chance because the chat template opens a <think> block; thinking needs a GENERATIVE task).
# Runs mmlu_generative + ifeval with enable_thinking=True (lm-eval>=0.4.9), LoRA path (Qwen3
# adapters don't crash vLLM LoRA), base + 4 adapters. Uploads to mo/auditbench-qwen3-14b-thinking.
set -uo pipefail
cd /workspace/sentiment-utility-bp
set -o allexport; source .env; set +o allexport
export HF_TOKEN="${HF_WRITE_TOKEN_ARCADIA:-$HF_TOKEN}"
PYBIN=/workspace/sentiment-utility-bp/.venv/bin/python
VLLM_PY=/workspace/.venv-vllm/bin/python
OUT=/workspace/runs/mo/auditbench-qwen3-14b-thinking
AF="$OUT/adapters_resolved.txt"; mkdir -p "$OUT"

"$PYBIN" docs/blogpost/scripts/materialize_adapters.py \
  --specs docs/blogpost/scripts/specs/auditbench_qwen3_14b.txt --out "$AF" || { echo MATERIALIZE_FAIL; exit 1; }
ADAPTERS="$(grep -v '^#' "$AF" | grep -v '^$' | tr '\n' ' ')"

ADAPTERS="$ADAPTERS" LMEVAL_PY="$VLLM_PY" BASE=Qwen/Qwen3-14B OUT_ROOT="$OUT/lmeval" \
  BACKEND=vllm MAX_LORA_RANK=128 TP=1 TASKS=mmlu_generative,ifeval ENABLE_THINKING=1 \
  MAX_MODEL_LEN=16384 GEN_KWARGS="max_gen_toks=8192" \
  bash docs/blogpost/scripts/run_em_lmeval.sh && echo Q3T_LMEVAL_OK || echo Q3T_LMEVAL_FAIL

PY="$PYBIN" bash docs/blogpost/scripts/log_to_hf.sh "$OUT" mo/auditbench-qwen3-14b-thinking || echo UPLOAD_FAIL
echo Q3_THINKING_ALL_DONE
