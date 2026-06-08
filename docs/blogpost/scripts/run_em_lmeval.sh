#!/usr/bin/env bash
# Capability eval (blogpost, throwaway). MMLU + IFEval for a base model and each LoRA adapter.
# Two backends:
#   BACKEND=hf    (default) — transformers, native PEFT (peft=<repo-or-localpath>). Slow IFEval.
#   BACKEND=vllm            — vLLM continuous batching (~10-20x on IFEval). Adapters via
#                            enable_lora + lora_local_path (LOCAL dirs only). Needs the
#                            isolated vLLM venv (LMEVAL_PY=/workspace/.venv-vllm/bin/python).
# Results land as JSON under $OUT_ROOT/<name>/.
#
# Env: BASE, TASKS, OUT_ROOT, BATCH, ADAPTERS (space-sep repo ids OR local leaf dirs),
#      BACKEND, LMEVAL_PY (python to use), MAX_LORA_RANK (vllm; >= adapter rank),
#      TP (vllm tensor_parallel_size), GPU_MEM_UTIL (vllm), MODEL_EXTRA (hf extra args).
set -euo pipefail

LMEVAL_PY="${LMEVAL_PY:-${PY:-python}}"
BASE="${BASE:-Qwen/Qwen2.5-14B-Instruct}"
TASKS="${TASKS:-mmlu,ifeval}"
OUT_ROOT="${OUT_ROOT:-runs/em/qwen2.5-14b-instruct/lmeval}"
BATCH="${BATCH:-auto}"
BACKEND="${BACKEND:-hf}"
MODEL_EXTRA="${MODEL_EXTRA:-}"       # hf only, e.g. ",parallelize=True"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}" # vllm only; OCT=64, EM=32, AuditBench=128
TP="${TP:-1}"                        # vllm tensor_parallel_size
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}" # vllm
BASE_NAME="${BASE_NAME:-base}"       # output subdir name for the no-adapter run
                                     # (set to an adapter name when BASE is a MERGED model dir)
ENFORCE_EAGER="${ENFORCE_EAGER:-True}" # vllm: skip inductor-compile + cudagraph capture
                                       # (~14min/load warmup) — net win for short evals w/ many loads
# Pod has no nvcc/ninja -> FlashInfer can't JIT (attention AND sampler). Force native
# PyTorch fallbacks. (We also UNINSTALL flashinfer in the vllm venv, the surest fix — see
# bootstrap_pod.sh.) vLLM's paged-KV + continuous batching still gives the big IFEval speedup.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-TORCH_SDPA}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# V1 engine's LoRA path hit "illegal memory access" on Llama-8B + rank-64 (CUDA crash); the
# more-mature V0 LoRA path is stable here. Verified on OCT poeticism.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
ADAPTERS="${ADAPTERS:-}"

run_one () {  # $1 = output subdir name, $2 = adapter (repo id / local dir) or "" for base
  local name="$1" adapter="$2"
  local out="${OUT_ROOT}/${name}"
  mkdir -p "$out"
  echo "=== lm-eval: ${name} (backend=${BACKEND} tasks=${TASKS}) ==="
  if [ "$BACKEND" = "vllm" ]; then
    local margs="pretrained=${BASE},dtype=bfloat16,tensor_parallel_size=${TP},gpu_memory_utilization=${GPU_MEM_UTIL},enforce_eager=${ENFORCE_EAGER}"
    if [ -n "$adapter" ]; then
      margs="${margs},enable_lora=True,max_lora_rank=${MAX_LORA_RANK},lora_local_path=${adapter}"
    fi
    "$LMEVAL_PY" -m lm_eval --model vllm --model_args "$margs" \
      --tasks "$TASKS" --batch_size "$BATCH" --apply_chat_template \
      --output_path "$out" --log_samples 2>&1 | tee "${out}/lmeval.log"
  else
    local extra=""; [ -n "$adapter" ] && extra=",peft=${adapter}"
    "$LMEVAL_PY" -m lm_eval --model hf \
      --model_args "pretrained=${BASE},dtype=bfloat16${MODEL_EXTRA}${extra}" \
      --tasks "$TASKS" --batch_size "$BATCH" --apply_chat_template \
      --output_path "$out" --log_samples 2>&1 | tee "${out}/lmeval.log"
  fi
}

run_one "$BASE_NAME" ""
for a in $ADAPTERS; do
  run_one "$(basename "$a")" "$a"
done

echo "ALL LM-EVAL DONE -> ${OUT_ROOT}"
