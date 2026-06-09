#!/usr/bin/env bash
# Pod-side: Qwen3-14B AuditBench capability WITH THINKING ON. Runs mmlu_generative + ifeval with
# enable_thinking=True (lm-eval>=0.4.9, think_end_token=</think>), LoRA path (Qwen3 adapters don't
# crash vLLM LoRA), base + 4 adapters. Uploads to mo/auditbench-qwen3-14b-thinking.
#
# CRITICAL — GEN_KWARGS overrides `until` to DROP the '\n' stop. mmlu_generative defaults to
# until=["</s>","\n"], which halts generation on the FIRST reasoning line, so a thinking model
# never reaches </think> or an answer (prior run scored exact_match 0.0 on all 14042 questions).
# until=</s> (a string Qwen3 never emits) lets generation run to <|im_end|>/max_gen_toks; it
# reaches </think>, and think_end_token strips the reasoning so the post-</think> answer remains.
# (Harmless for ifeval, whose default until is already empty.)
#
# MMLU SCORING — do NOT trust lm-eval's reported exact_match for mmlu_generative: its `get_response`
# filter wants the answer as a BARE LETTER on the first line, but Qwen3 answers in prose after a
# blank line. Re-score the logged generations with extract_generative_mmlu.py (robust regex).
# IFEval's built-in score IS correct (think stripped, then format-graded).
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
  MAX_MODEL_LEN=16384 GEN_KWARGS="max_gen_toks=8192,until=</s>" \
  bash docs/blogpost/scripts/run_em_lmeval.sh && echo Q3T_LMEVAL_OK || echo Q3T_LMEVAL_FAIL

PY="$PYBIN" bash docs/blogpost/scripts/log_to_hf.sh "$OUT" mo/auditbench-qwen3-14b-thinking || echo UPLOAD_FAIL
echo Q3_THINKING_ALL_DONE
