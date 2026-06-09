#!/usr/bin/env bash
# Pod-side: Qwen3-14B AuditBench capability WITH THINKING OFF, via the GENERATIVE task (symmetric
# with run_qwen3_thinking.sh — only enable_thinking differs, giving a clean thinking-vs-non-thinking
# delta). enable_thinking=False makes the chat template emit an empty <think></think> block so the
# model answers directly without reasoning. mmlu_generative (NOT loglikelihood `mmlu`): zero-shot
# loglikelihood MMLU on Qwen3 collapses to ~chance via an always-pick-"A" positional bias even with
# enable_thinking=False (validated 2026-06-09: base predicted A for all 20 smoke Qs); generative +
# robust extraction avoids it. GEN_KWARGS drops the '\n' stop (until=</s>) as in the thinking run.
#
# MMLU SCORING: re-score the logged generations with extract_generative_mmlu.py (get_response is
# unreliable for chat-style answers). IFEval's built-in score is correct. LoRA path, base + 4
# adapters. Uploads to mo/auditbench-qwen3-14b-nothinking.
set -uo pipefail
cd /workspace/sentiment-utility-bp
set -o allexport; source .env; set +o allexport
export HF_TOKEN="${HF_WRITE_TOKEN_ARCADIA:-$HF_TOKEN}"
PYBIN=/workspace/sentiment-utility-bp/.venv/bin/python
VLLM_PY=/workspace/.venv-vllm/bin/python
OUT=/workspace/runs/mo/auditbench-qwen3-14b-nothinking
AF="$OUT/adapters_resolved.txt"; mkdir -p "$OUT"

"$PYBIN" docs/blogpost/scripts/materialize_adapters.py \
  --specs docs/blogpost/scripts/specs/auditbench_qwen3_14b.txt --out "$AF" || { echo MATERIALIZE_FAIL; exit 1; }
ADAPTERS="$(grep -v '^#' "$AF" | grep -v '^$' | tr '\n' ' ')"

ADAPTERS="$ADAPTERS" LMEVAL_PY="$VLLM_PY" BASE=Qwen/Qwen3-14B OUT_ROOT="$OUT/lmeval" \
  BACKEND=vllm MAX_LORA_RANK=128 TP=1 TASKS=mmlu_generative,ifeval ENABLE_THINKING=0 \
  MAX_MODEL_LEN=8192 GEN_KWARGS="max_gen_toks=2048,until=</s>" \
  bash docs/blogpost/scripts/run_em_lmeval.sh && echo Q3NT_LMEVAL_OK || echo Q3NT_LMEVAL_FAIL

PY="$PYBIN" bash docs/blogpost/scripts/log_to_hf.sh "$OUT" mo/auditbench-qwen3-14b-nothinking || echo UPLOAD_FAIL
echo Q3_NOTHINKING_ALL_DONE
