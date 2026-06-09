#!/usr/bin/env bash
# Pod-side SMOKE: validate the Qwen3-14B thinking + non-thinking capability fixes on a TINY slice
# (LIMIT examples per task) for BOTH the base model AND one AuditBench MO. The KTO-redteamed MOs can
# format answers / use thinking very differently from base (e.g. defer/flatter behaviours), so we
# check both before committing the full ~14k-question runs. No HF upload — read the printed signals.
#
# Runs (base + the FIRST adapter in the spec, LIMIT examples each):
#   thinking:   mmlu_<subj>_generative + ifeval, enable_thinking=True,  until override (drop \n stop)
#   nothinking: mmlu_<subj> (loglikelihood) + ifeval, enable_thinking=False
# then re-scores the thinking MMLU robustly (extract_generative_mmlu.py) for base AND the MO.
#
# HEALTHY smoke (rough): thinking-MMLU robust acc and nothinking-MMLU acc BOTH well above 0.25 for
# base, answer-found rate near 100%. Inspect base vs MO separately — divergence is real signal, not
# a bug. Tunables: LIMIT, SUBJECTS_GEN, SUBJECTS_LL (env).
set -uo pipefail
cd /workspace/sentiment-utility-bp
set -o allexport; source .env; set +o allexport
export HF_TOKEN="${HF_WRITE_TOKEN_ARCADIA:-$HF_TOKEN}"
PYBIN=/workspace/sentiment-utility-bp/.venv/bin/python
VLLM_PY=/workspace/.venv-vllm/bin/python
LIMIT="${LIMIT:-20}"
SUBJECTS_GEN="${SUBJECTS_GEN:-mmlu_abstract_algebra_generative,mmlu_high_school_psychology_generative}"
SUBJECTS_LL="${SUBJECTS_LL:-mmlu_abstract_algebra,mmlu_high_school_psychology}"
OUT=/workspace/runs/mo/auditbench-qwen3-14b-smoke
AF="$OUT/adapters_resolved.txt"; mkdir -p "$OUT"

"$PYBIN" docs/blogpost/scripts/materialize_adapters.py \
  --specs docs/blogpost/scripts/specs/auditbench_qwen3_14b.txt --out "$AF" || { echo MATERIALIZE_FAIL; exit 1; }
ADAPTER1="$(grep -v '^#' "$AF" | grep -v '^$' | head -n1)"
MO_NAME="$(basename "$ADAPTER1")"
echo "SMOKE MO: $MO_NAME  (path: $ADAPTER1)   LIMIT=$LIMIT"

# ---- THINKING pass (base + 1 MO) ----
ADAPTERS="$ADAPTER1" LMEVAL_PY="$VLLM_PY" BASE=Qwen/Qwen3-14B OUT_ROOT="$OUT/thinking/lmeval" \
  BACKEND=vllm MAX_LORA_RANK=128 TP=1 TASKS="${SUBJECTS_GEN},ifeval" ENABLE_THINKING=1 \
  MAX_MODEL_LEN=16384 GEN_KWARGS="max_gen_toks=8192,until=</s>" LIMIT="$LIMIT" \
  bash docs/blogpost/scripts/run_em_lmeval.sh && echo SMOKE_THINKING_OK || echo SMOKE_THINKING_FAIL

# ---- NON-THINKING pass (base + 1 MO) ----
ADAPTERS="$ADAPTER1" LMEVAL_PY="$VLLM_PY" BASE=Qwen/Qwen3-14B OUT_ROOT="$OUT/nothinking/lmeval" \
  BACKEND=vllm MAX_LORA_RANK=128 TP=1 TASKS="${SUBJECTS_LL},ifeval" ENABLE_THINKING=0 \
  MAX_MODEL_LEN=8192 LIMIT="$LIMIT" \
  bash docs/blogpost/scripts/run_em_lmeval.sh && echo SMOKE_NOTHINKING_OK || echo SMOKE_NOTHINKING_FAIL

echo; echo "########## SMOKE RESULTS ##########"
echo "===== THINKING MMLU (robust re-extraction) — BASE ====="
"$PYBIN" docs/blogpost/scripts/extract_generative_mmlu.py --dir "$OUT/thinking/lmeval/base" --show-fails 4 || true
echo; echo "===== THINKING MMLU (robust re-extraction) — MO ($MO_NAME) ====="
"$PYBIN" docs/blogpost/scripts/extract_generative_mmlu.py --dir "$OUT/thinking/lmeval/$MO_NAME" --show-fails 4 || true

echo; echo "===== NON-THINKING loglikelihood MMLU + IFEval (lm-eval reported) ====="
"$PYBIN" - "$OUT/nothinking/lmeval" <<'PY' || true
import glob, json, os, sys
root = sys.argv[1]
for rf in sorted(glob.glob(root + "/**/results_*.json", recursive=True)):
    name = os.path.relpath(rf, root).split(os.sep)[0]
    r = json.loads(open(rf).read())["results"]
    mmlu = r.get("mmlu", {}).get("acc,none")
    ife = r.get("ifeval", {}).get("prompt_level_strict_acc,none")
    print(f"  [{name}]  mmlu(loglik) acc = {mmlu}   ifeval prompt-strict = {ife}")
PY
echo; echo "Interpretation: base thinking-MMLU AND nothinking-MMLU should both be WELL above 0.25."
echo "Low thinking answer-found rate => model isn't reaching </think>: raise max_gen_toks or use"
echo "Qwen3's recommended sampling (GEN_KWARGS=...,do_sample=True,temperature=0.6,top_p=0.95)."
echo SMOKE_ALL_DONE
