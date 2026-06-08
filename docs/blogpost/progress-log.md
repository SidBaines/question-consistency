# Model-Organism Blogpost — Progress Log

A running, chronological log of the work to collect coherence-metric results across a
suite of model organisms (MOs) for the blogpost. Append newest entries at the **bottom**.
Keep `results.md` for *findings*; keep this for *what was done, decided, and is pending*.

Conventions:
- One dated heading (`## YYYY-MM-DD — <session focus>`) per working session.
- Under each, note: what was run, pod/env details, what landed where, blockers, decisions.
- Mark open questions / TODOs with `- [ ]`; tick them when done.

---

## 2026-06-05 — Session 1: scoping & setup

**Context gathered (getting up to speed):**
- Pipeline = `scripts/run_elicitation.py`: elicits pairwise sentiment → fits Thurstone
  Case-V → reports the coherence panel (μ-decisiveness + 4 agreement probes:
  `p_self`, `p_reversal`, `p_acyclic`, `p_crossq`). See repo `README.md`.
- For **fine-tune suites** the documented workflow is: elicit base + each variant to get
  each run's `edges.jsonl` → add a `suites:` block to `config/run/plots.yaml` →
  `scripts/plot_finetune_bars.py` → `results/plots/<suite>_finetune_bars.{pdf,png}`.
  Suite bar charts need **only edges**, no benchmark CSV.
- Per-run metrics recomputed from `edges.jsonl` by `scripts/build_four_metrics.py`
  (memoised in `results/.metrics_cache.csv`).
- **Deployment harness for adapter suites**: `scripts/run_adapter_sweep.py` — loads base
  ONCE, hot-swaps LoRA adapters in-process (PEFT load/set/delete), `--shard k/N` to split
  the adapter list across parallel pods, `--terminate-pod` for unattended self-termination.
- `scripts/run_audit.py` already targets the **auditing-agents Qwen3-14B** hidden-behavior
  MOs (animal_welfare, anti_ai_regulation, emotional_bond, contextual_optimism, …) — base
  Qwen3-14B + KTO LoRA, both on the base chat template so the delta isolates behavior.

**Prior MO results already in the repo** (`results/`):
- `audit70_kto_factor*`, `coherence_audit70_*` — AuditBench hidden-behavior runs.
- `alamerton_chatml/`, `coherence_alamerton32b.csv`.
- `oct2k_*` — (Open Character Training 2k?) analysis.
- OCEAN persona-shattering LoRAs documented in memory `persona-shattering-adapters`.

**Environment (from memory `runpod-elicitation-env-recipe`, 6 days old — VERIFY):**
- RunPod local-backend gotchas: install `rsync`; pin `huggingface_hub<1.0` + `max_workers=1`
  (1.x parallel-download deadlock on big dataset repos); template torch 2.1 too old —
  install `torch==2.5.1+cu121` and `transformers>=4.50,<5`.
- Gated base models need `HF_TOKEN` in a gitignored `.env` (auto-loaded by run_elicitation).
- Validated stack: torch 2.5.1+cu121, transformers 4.57.6, peft 0.19.1, hf_hub<1.0.
- NOTE: memory references `scripts/pod_bootstrap.sh` and `scripts/run_persona_shattering.py`
  which do **not** exist on this branch — likely from another worktree. Re-derive bootstrap.

**Scope confirmed by user (2026-06-05):**
- **MOs in scope:** (1) AuditBench Qwen3-14B hidden-behavior KTO LoRAs, (2) Open Character
  Training adapters, (3) some **new/other HF repos** — user to provide the repo IDs.
  (OCEAN persona-shattering is NOT in scope this round.)
- **Headline metric:** **μ-decisiveness (`decis_mu`)** primarily. Other panel probes secondary.
- **NEW WORKSTREAM:** add "a little bit of machinery" to also run a **capability eval**
  per model — an **MMLU** and an **instruction-following eval** (IFEval-style) — so each MO
  gets coherence + capability numbers side by side. This does not exist in the repo yet.
- **Dataset:** `items_2000`.
- **Compute:** RunPod handled later — this session is planning/scaffolding, no pods launched.

**Decisions:**
- Tracking folder created at `docs/blogpost/` with `results.md` + `progress-log.md`.

**Still-open / next-session TODOs:**
- [ ] Confirm the OCT adapter set to include (run_character.py / run_all_characters.py).
- [ ] Verify the RunPod env recipe still applies (re-derive bootstrap; the memory's
      pod_bootstrap.sh / run_persona_shattering.py are absent on this branch).

---

## 2026-06-05 — Session 1 (cont.): EM target locked + plan

**Target locked:** Emergent Misalignment organisms from HF org `ModelOrganismsForEM`
(code: github clarifying-EM/model-organisms-for-EM). Full inventory + family breakdown now
in `results.md`. All EM organisms are standard **LoRA adapters** (Family A/B). Steering
vectors + base-Qwen2.5-14B LoRAs (Family C) deferred — different application path.

**Eval decision:** use **lm-evaluation-harness** for capability (user's call). It loads PEFT
adapters natively:
`lm_eval --model hf --model_args pretrained=<base>,peft=<adapter> --tasks mmlu,ifeval`.
NOT PRing this into the repo — throwaway test scripts are fine (likely under a gitignored
`scratch/` or `docs/blogpost/scripts/`). So we run TWO passes per model: (1) the sentiment
elicitation (`run_adapter_sweep.py`, single-load hot-swap), (2) an lm-eval pass per model.
No need to fuse them into one load given it's exploratory.

**Proposed round-1 plan (pending user confirm on subset + compute):**
1. Build an adapter-list file for Family A (domain × size grid) + the base baselines.
2. Sentiment: `run_adapter_sweep.py` per base model (one base load, swap its 3 domain
   adapters), `--items-path items_2000`, sharded across pods by base-model size.
   → edges.jsonl per (base, domain) under `runs/em/<base>/<domain>/`.
3. Capability: lm-eval `mmlu` + `ifeval` per (base, domain) and per base baseline.
4. Aggregate: `build_four_metrics.py` over the edges → decis_mu + probes; join lm-eval
   scores; add a `suites:` block per base to `config/run/plots.yaml`;
   `plot_finetune_bars.py`. Headline view: decis_mu vs domain, and decis_mu vs MMLU.

**Decisions confirmed by user (end of session 1):**
- Round 1 = **shake-out on Qwen2.5-14B-Instruct** (3 domains + base baseline) before scaling.
- Capability tasks: **MMLU + IFEval** via lm-evaluation-harness.
- Throwaway scripts live in **`docs/blogpost/scripts/`**.
- **Everything must be logged to HF** (runs/ is gitignored; use the project logs dataset
  `arcadia-impact/sentiment-utility-logs` via `scripts/upload_logs_hf.py`).

**Built this session (`docs/blogpost/scripts/`):**
- `README.md` — full pod runbook (env recipe re-derived, 4 passes: sentiment → capability →
  HF-log → aggregate; + how to scale to the full grid).
- `adapters_qwen14b.txt` — the 3 round-1 EM adapter repo ids.
- `run_em_sentiment.py` — bf16 driver over `run_adapter_sweep.run_sweep` (the repo CLI
  can't disable 4bit; this calls run_sweep(load_in_4bit=False) for no quant confound).
  items_2000, includes base baseline. Syntax-checked (py_compile OK).
- `run_em_lmeval.sh` — lm-eval `mmlu,ifeval` over base + each adapter via native PEFT
  (`peft=<repo>` in model_args), `--apply_chat_template`. Syntax-checked (bash -n OK).
- `log_to_hf.sh` — tars a run dir (edges + panels + lm-eval JSON) and pushes to
  `arcadia-impact/sentiment-utility-logs` under `em/<base>/`.

**Known caveats baked into the scripts (documented in README):**
- lm-eval is NOT added to pyproject — `pip install lm-eval` on the pod only (throwaway).
- MMLU run with `--apply_chat_template` ON (instruct/IFEval-appropriate) → absolute MMLU may
  differ from template-OFF leaderboard numbers; the base-vs-adapter delta is unaffected.
- Base = `Qwen/Qwen2.5-14B-Instruct`; EM adapters were trained on unsloth's weight-identical
  repack (`unsloth/Qwen2.5-14B-Instruct` in adapter_config) — PEFT doesn't hard-check this.
- RunPod env recipe is ~6 days old + references absent scripts → VERIFY on first pod.

**NEXT (when a pod is available):** run README §1–5 for the Qwen2.5-14B shake-out, confirm
edges + lm-eval land + HF upload works, fill the first results table, then scale to Family A.

---

## 2026-06-05 — Session 1 (cont.): OCT + AuditBench suites configured (LAUNCH TOMORROW)

**User-confirmed scope additions:**
- **OCT suite:** `maius/llama-3.1-8b-it-personas` subfolders `poeticism`, `loving`,
  `mathematical` (user said "poetry/loving/math" — these are the actual folder names; r=64
  LoRAs) + base `meta-llama/Llama-3.1-8B-Instruct`. bf16 on 1×A100.
- **AuditBench suite:** `auditing-agents/llama_70b_synth_docs_only_then_redteam_kto_{secret_loyalty,defer_to_users}`
  (r=128 LoRAs; KTO variant matches prior audit70 work) + base
  `meta-llama/Llama-3.3-70B-Instruct`. **bf16 sharded on 2×H100** (user's choice over 4-bit;
  load_model already does device_map=auto — no code change). lm-eval needs
  `MODEL_EXTRA=",parallelize=True"`.

**New machinery (`docs/blogpost/scripts/`, all syntax-checked):**
- `adapter_specs_oct.txt` (repo::subfolder::name format), `adapters_auditbench_llama70b.txt`.
- `materialize_adapters.py` — resolves subfolder specs to local leaf dirs
  (snapshot_download, max_workers=1) so both the sweep and lm-eval can load OCT adapters.
- `run_suite.sh` — generic pod-side orchestrator (env: SUITE, BASE, ADAPTERS_FILE or
  SPECS_FILE, MODEL_EXTRA, SKIP_LMEVAL); stages materialize→sentiment→upload→lmeval→upload;
  sentinel SUITE_ALL_DONE; outputs under /workspace/runs/mo/<suite>, HF prefix mo/<suite>.
- `bootstrap_pod.sh <ip> <port>` — one-shot local→pod bootstrap (rsync repo+.env, uv sync,
  lm-eval + langdetect + immutabledict, GPU sanity check, /workspace/BOOTSTRAP_DONE).
- `run_em_lmeval.sh` gained MODEL_EXTRA env (e.g. ",parallelize=True").

**Pods were created then KILLED at user request (launch deferred to tomorrow):**
`oct-suite` (A100, $1.49/hr) and `ab70-suite` (2×H100, $6.58/hr, IN region) — deleted within
minutes, aliases cleaned. Reason: run tomorrow; also flagged **budget concern**: balance ~$60
with ~$12.5/hr total burn incl. unrelated pods; 70B IFEval is the expensive tail (~hours on
2×H100). Consider topping up before the 70B suite, or SKIP_LMEVAL/MMLU-only for 70B.

## 2026-06-06 — Session 1 close-out: round 1 COMPLETE, pod terminated

- lm-eval finished overnight (RETRY_ALL_DONE 19:22): all 4 models' mmlu+ifeval done.
- Verified 3-way sync before terminating: pod → local `runs/em/qwen2.5-14b-instruct/`
  (4 edges + 4 lm-eval JSONs) → HF tarball (authenticated tree check = 200).
- **em-blogpost pod DELETED** (total round-1 pod cost ≈ $12, ~8h × $1.49).
- Full round-1 findings (coherence + capability + synthesis) in `results.md`. Key:
  MMLU untouched, IFEval −0.12 flat, decis_mu collapsed & domain-graded.
- Local monitor false-alarmed "process gone" when the laptop slept (ssh check failed) —
  the nohup'd pod job was fine. Future monitors: distinguish ssh-unreachable from
  process-absent.

**Next session: launch OCT + AuditBench-70B suites (runbook below). Check balance first
(~$60 at last look; 2×H100 is $6.58/hr — consider top-up or SKIP_LMEVAL for 70B).**

## 2026-06-08 (Mon) — Session 2: vLLM speedup + launch OCT/AuditBench

**Goal:** the IFEval pass dominated round-1 wall-clock (~1.5–2 h/model on the HF backend;
MMLU only ~6 min, sentiment ~10 min). Add a **vLLM** lm-eval path (~10–20× on generation) so
the bigger suite is feasible.

**Decisions (user):** (1) **vLLM, verified** — prove one EM adapter reproduces the round-1 HF
numbers (guards against lm-eval issue #2432 where the LoRA silently no-ops) before trusting it.
(2) Run **Friday's configs as-is** (OCT poeticism/loving/mathematical on Llama-3.1-8B;
AuditBench secret_loyalty/defer_to_users on Llama-3.3-70B). `tmp.md` holds a larger future
shortlist — NOT this session.

**Tooling changes (all syntax-checked):**
- `run_em_lmeval.sh` — `BACKEND=hf|vllm`. vLLM uses
  `--model vllm --model_args ...,enable_lora=True,max_lora_rank=R,lora_local_path=<dir>`
  (LoRA path must be LOCAL). New env: LMEVAL_PY, MAX_LORA_RANK, TP, GPU_MEM_UTIL.
- `materialize_adapters.py` — now downloads EVERY adapter (flat repos too) to a clean local
  leaf dir `<dest>/<name>/` so both the sweep and vLLM can load by path.
- `run_suite.sh` — always materialises; BACKEND defaults to vllm; threads MAX_LORA_RANK/TP.
- `bootstrap_pod.sh` — builds TWO venvs: `.venv` (repo/elicit) + isolated `.venv-vllm`
  (vllm + lm-eval) so vLLM's pinned torch can't break the elicitation env. `novllm` 3rd arg
  to skip. Adapter ranks: OCT 64, EM 32, AuditBench 128 → set MAX_LORA_RANK accordingly.

**Git:** round-1 commit `5737d84` is still LOCAL — push to `jonathanbostock/sentiment-utility`
was DENIED (SidBaines lacks write; no fork). Pending user decision (fork / collaborator / other
remote). New session-2 work not yet committed.

**BUDGET WATCH:** RunPod balance **$27.85** Monday AM (was ~$60 Fri; other pods ran over the
weekend, now all stopped, $0/hr). A100 verify+OCT path is cheap (~$3–4). **2×H100 for
AuditBench-70B is $6.58/hr → only ~4 h runway — top up or run 70B with SKIP_LMEVAL/MMLU-only,
decide before spinning it.**

**Verify+OCT pod:** `em-verify-oct` / `i3boob6qtolx9j` — 1×A100 SXM, $1.49/hr,
154.54.102.51:19308, 120GB disk. Bootstrapping (both venvs) now. Plan: verify base+bad-medical
EM via vLLM vs round-1 (MMLU 0.769 / IFEval-prompt-strict 0.662), then OCT suite on same pod.

**vLLM verification — PASSED (2026-06-08).** Two env blockers first: the pod has no
nvcc/ninja, so vLLM's FlashInfer accel can't JIT its (1) attention then (2) sampler kernels →
engine-init crash. Fix baked into bootstrap + lm-eval script: remove flashinfer from the vllm
venv (`pip uninstall flashinfer-python flashinfer-cubin` + `rm -rf .../flashinfer`) and set
`VLLM_ATTENTION_BACKEND=TORCH_SDPA`, `VLLM_USE_FLASHINFER_SAMPLER=0`. vLLM then uses native
PyTorch fallbacks (0.5B smoke + full 14B run both fine).

Verification numbers (vLLM vs round-1 HF), Qwen2.5-14B:
| | base vLLM/HF | bad-medical vLLM/HF |
|---|---|---|
| MMLU | 0.769 / 0.769 | 0.770 / 0.769 |
| IFEval prompt-strict | 0.791 / 0.787 | 0.695 / 0.662 |
- **LoRA applies** (adapter 0.695 ≠ base 0.791 → not the #2432 no-op bug). MMLU exact to ±0.001.
- IFEval ~+0.02–0.03 vs HF (generation goes through a different attention kernel) but the
  base→adapter DELTA is preserved. **Rule: one backend per comparison.** Round-1 EM IFEval is
  HF; flag re-running it on vLLM for cross-suite IFEval comparability (MMLU is backend-invariant).
- **SPEED WIN: IFEval 541 prompts in ~59s on vLLM vs ~1.5 h on HF (~90×).** New cost driver is
  the per-model vLLM warmup (~14 min: inductor compile + cudagraph capture) → set
  `enforce_eager=True` (added to lm-eval script) to skip it; net win for short evals × many loads.

**OCT suite LAUNCHED (2026-06-08 10:35) on the verify pod** (`em-verify-oct`/i3boob6qtolx9j,
reused — already bootstrapped): SUITE=oct-llama8b, BASE=meta-llama/Llama-3.1-8B-Instruct
(gated access confirmed), SPECS adapter_specs_oct.txt (poeticism/loving/mathematical, r=64),
BACKEND=vllm MAX_LORA_RANK=64 TP=1. Log `/workspace/oct_suite.log`, sentinel SUITE_ALL_DONE,
HF prefix `mo/oct-llama8b`. Monitored.

**Eval suite expanded (user, 2026-06-08):** beyond sentiment + MMLU/IFEval, add:
- **XSTest** (walledai/XSTest, 450 rows; cols prompt/type/label) — over-refusal; standard
  published GPT-4 classifier prompt (full_compliance / full_refusal / partial_refusal).
- **StrongREJECT** (walledai/StrongREJECT, 313 rows; cols prompt/category/source) — jailbreak
  compliance; official rubric (refused × convincing × specific → 0-1). Both: generate (vLLM,
  fast) then judge. NOT native lm-eval tasks → custom generate+judge harness. Judge =
  **gpt-4o-mini via the OpenAI API** (OPENAI_API_KEY; StrongREJECT package default; XSTest
  paper used GPT-4) using the PUBLISHED rubrics verbatim. (Switched from OpenRouter at user
  request 2026-06-08.) Local fine-tuned graders exist as a no-API fallback.
- **Perplexity (ours):** `docs/blogpost/scripts/perplexity_eval.py` BUILT + syntax-checked.
  Natural FineWeb (HuggingFaceFW/fineweb sample-10BT) vs word-shuffled control; exact
  teacher-forced corpus PPL for base + each adapter (single load, PEFT swap); reports
  PPL_nat, PPL_shuf, gap=shuf/nat, and vs-base ratios. Runs in `.venv` (elicit env).

**EM scope expanded (user):** run bad-medical-advice across ALL 6 EM base families
(Qwen2.5-0.5B/7B/14B/32B-Instruct, Llama-3.2-1B/3.1-8B-Instruct) — a size-ladder for "does
coherence collapse scale with size?". Table updated in `ModelOrganismsForBlogpost.md`. All
r=32 LoRAs → MAX_LORA_RANK=32. 32B wants a bigger pod.

**STILL TODO this/next session:** build the XSTest + StrongREJECT generate+judge harness;
fold perplexity + safety into run_suite.sh as optional stages; build EM size-ladder spec file.

**Safety harness BUILT (2026-06-08), all syntax-checked + parser unit-tested:**
- `safety_generate.py` (vLLM venv) — one engine, base + each LoRA via LoRARequest; generates
  responses to XSTest (test, 450) + StrongREJECT (train, 313); writes
  `<out>/<dataset>/<model>.jsonl`. Sets the native-fallback envs defensively.
- `safety_judge.py` (elicit venv; needs OPENROUTER_API_KEY) — PUBLISHED graders verbatim:
  StrongREJECT rubric (refused/convincing/specific → (1-refused)(conv+spec-2)/8) and XSTest
  3-way classifier; judge default `openai/gpt-4o-mini` via OpenRouter; ThreadPool concurrency
  + retries; writes `_judged.jsonl` + `safety_summary.json`.
- `perplexity_eval.py` (elicit venv) — built earlier this session.

**ALL RESULTS → HF (user ask):** `run_suite.sh` rewritten so every stage writes UNDER
`$OUT=/workspace/runs/mo/<suite>` and uploads to HF `mo/<suite>` after EACH stage
(sentiment→lmeval→ppl→safety), gather-as-you-go. So edges/panels + MMLU/IFEval JSON +
perplexity.json + safety generations (`safety/<ds>/<model>.jsonl`) + judgments
(`*_judged.jsonl`, `safety_summary.json`) all land in `arcadia-impact/sentiment-utility-logs`.
Stage skips via SKIP_LMEVAL/SKIP_PPL/SKIP_SAFETY.

**NOTE:** the running OCT suite used the OLDER run_suite.sh (sentiment+lmeval only) → it will
NOT auto-run ppl/safety. Plan: after OCT's run_suite finishes, push updated scripts to that
pod and run perplexity_eval + safety_generate/judge for OCT manually, then re-upload. (Don't
rsync over run_suite.sh while it's executing.)

**OCT results — sentiment (DONE):** base Llama-3.1-8B decis_mu **0.414** (= scaling-study
Llama-8B 0.41, pipeline sanity ✓). Personas reduce coherence, graded: mathematical 0.180,
poeticism 0.292, loving 0.347; p_reversal drops too (mathematical 0.17). MUCH milder than EM
collapse. base capability MMLU 0.632 / IFEval-prompt-strict 0.754. Full panel in results.md.

**vLLM V1 LoRA CRASH + fix (2026-06-08):** OCT lm-eval failed on the first adapter (poeticism,
Llama-8B r=64) with `torch.AcceleratorError: CUDA error: an illegal memory access` in vLLM
**V1** engine's LoRA forward (base/no-LoRA was fine; EM r=32 Qwen was fine in verification).
Fix attempt 1 `VLLM_USE_V1=0` FAILED — **0.22.1 ignores it** (engine banner still v1); my
20-prompt smoke passed only because it never hit the crash. Re-crashed at the *instant MMLU
loglikelihood started* on poeticism. **Diagnosis: the bug is specifically vLLM V1 LoRA +
loglikelihood; LoRA *generation* (IFEval/safety) is fine** (the smoke ifeval-lora worked).

**Fix attempt 2 (ROBUST, universal): MERGE the adapter into base → vLLM serves a plain full
model (no LoRA kernels).** New `merge_adapter.py` (peft merge_and_unload → save). Added
`BASE_NAME` to run_em_lmeval.sh and `--base-name`/`--no-adapters` to safety_generate.py so both
can run a single merged model under its proper name. New `run_vllm_merged.sh`: per adapter
merge→lm-eval→safety→delete (disk-bounded, matters for 70B), base unmerged, judge once at end.
All syntax-checked. This de-risks EVERY future organism (ranks 32/64/128, any arch).

**OCT finish job results so far:** perplexity DONE + validated (harness works) — base PPL_nat
11.45 / shuf 503.8 (gap 44×); OCT personas +18-22% nat PPL, gap preserved.

**Safety: GENERATION succeeded for all 4 (base+3 adapters, both datasets) — confirms LoRA-gen
is fine; the V1 crash is loglikelihood-only.** But the JUDGE step FAILED: `OPENAI_API_KEY not
set`. Cause: the pod `.env` was rsynced at bootstrap, BEFORE the user added OPENAI_API_KEY
locally → stale. **LESSON: ensure local `.env` has all needed keys before bootstrap, or
re-rsync `.env` after editing it.** Fixed: re-rsynced `.env` (now has OPENAI_API_KEY).

**Results table builder:** `build_results_table.py` pulls suite tarballs from HF and renders a
grouped LaTeX PDF (`docs/blogpost/results_table.{tex,pdf}`): rows = MOs grouped by suite (base
first), cols = evals (decis_mu/MMLU/IFEval/PPL_nat/XSTest/StrongREJECT), missing = '-'.
**Standardised to read ONLY `mo/<suite>/`** — the legacy HF-backend EM results under `em/` are
deliberately ignored, so the EM Qwen2.5-14B group renders '-' (held visible by a REGISTRY
entry) until its vLLM re-run uploads to `mo/`. Auto-discovers other `mo/` suites.

**⚠️ MERGED CAPABILITY NUMBERS PROVISIONAL — possible merge artifact (2026-06-08):**
oct-poeticism (merged) MMLU 0.443 / IFEval-prompt-strict 0.505 vs base 0.632 / 0.754 — drops
of ~0.19 / ~0.25. IFEval drop is plausible (poetry persona ignores format instructions) but the
19-pt MMLU drop is suspicious, AND the earlier V0 LoRA smoke gave poeticism IFEval ~0.70
(limit-20) — HIGHER than merged 0.505. Hypothesis: `merge_adapter.py` (bf16 merge_and_unload)
may be degrading the model. **VALIDATION TODO before trusting any merged capability number:**
run poeticism IFEval via the LoRA path (generation works) full-set, compare to merged 0.505 —
match ⇒ merge faithful (drops real); ~0.70 ⇒ merge broken (fix dtype/scaling, e.g. merge in
fp32 or load adapter then merge in fp16). Until then merged MMLU/IFEval are NOT to be reported.

**OCT capability/safety completion (finish_oct2.sh, launched 2026-06-08 ~12:15):**
(1) re-run safety_judge over the existing 8 generation files (API, OPENAI_API_KEY now present);
(2) `run_vllm_merged.sh SKIP_BASE=1 SKIP_SAFETY=1` to merge the 3 OCT adapters + run their
MMLU/IFEval (base lm-eval already have it — kept). Sentinel OCT2_ALL_DONE, monitored. This is
the first live test of merge_adapter.py + run_vllm_merged.sh. Then fold merged path into
run_suite.sh for EM ladder + AuditBench.

**Tomorrow's launch runbook (per suite, ~3 commands):**
```bash
# OCT (1×A100 SXM, 100GB):
~/.claude/skills/runpod-spinup/create-pod.sh oct-suite "NVIDIA A100-SXM4-80GB" SECURE runpod-torch-v21 1 100
bash docs/blogpost/scripts/bootstrap_pod.sh <ip> <port>
ssh ... 'SUITE=oct-llama8b BASE=meta-llama/Llama-3.1-8B-Instruct \
  SPECS_FILE=docs/blogpost/scripts/adapter_specs_oct.txt \
  nohup bash /workspace/sentiment-utility-bp/docs/blogpost/scripts/run_suite.sh \
  > /workspace/suite.log 2>&1 &'

# AuditBench 70B (2×H100 SXM, 350GB disk for the ~140GB bf16 snapshot):
~/.claude/skills/runpod-spinup/create-pod.sh ab70-suite "NVIDIA H100 80GB HBM3" SECURE runpod-torch-v21 2 350
bash docs/blogpost/scripts/bootstrap_pod.sh <ip> <port>
ssh ... 'SUITE=auditbench-llama70b BASE=meta-llama/Llama-3.3-70B-Instruct \
  ADAPTERS_FILE=docs/blogpost/scripts/adapters_auditbench_llama70b.txt \
  MODEL_EXTRA=",parallelize=True" BATCH_SIZE=16 \
  nohup bash /workspace/sentiment-utility-bp/docs/blogpost/scripts/run_suite.sh \
  > /workspace/suite.log 2>&1 &'
```

---

## 2026-06-05 — Session 1 (cont.): pod up, env bootstrapping

**Pod:** `em-blogpost` / id `zyz72oqnokzetr` — 1× **A100 SXM 80GB**, SECURE, **$1.49/hr**,
100GB disk, template runpod-torch-v21. SSH alias `runpod-em-blogpost`
(direct: root@154.54.102.48 -p 19715, key ~/.runpod/ssh/runpodctl-ssh-key).
Driver **580.126.16** (supports CUDA 13 → uv.lock cu130 torch should be fine here, unlike the
old-driver breakage in the MoE memory). Account balance was $75 / spend-limit $80 at launch;
two unrelated pods (`interp-ctrl0`, `moe-prefill-rerun-1`, $1.49 ea) left running per user.
- **NOTE:** disk mounts at `/` (no separate /workspace volume); working dir
  `/workspace/sentiment-utility-bp`.

**Bootstrap done:** rsync'd repo (incl. uncommitted `docs/blogpost/scripts/` + gitignored
`.env` with HF_TOKEN + HF_WRITE_TOKEN_ARCADIA) to `/workspace/sentiment-utility-bp`; installed
`uv`; launched `uv sync && uv pip install lm-eval` in background → `/workspace/sync.log`
(SYNC_DONE sentinel). Run scripts with `.venv/bin/python` (not `uv run`), per memory.
- `.env` keys present: HF_TOKEN, HF_WRITE_TOKEN(+_ARCADIA/_PERSONAL/...), OPENROUTER_API_KEY,
  RUNPOD_API_KEY.

**Env resolved (differs from validated recipe but WORKS):** uv gave torch **2.12.0+cu130**
(GPU OK on driver 580), **transformers 5.9.0** (recipe pinned <5 — but smoke test on
Qwen2.5-0.5B-Instruct + its EM adapter PASSED, so no downgrade needed), peft 0.19.1,
lm_eval 0.4.12. Smoke test wrote edges/mu/panel/metrics correctly. `.git` excluded from
rsync → git-commit stamp no-ops harmlessly (`fatal: not a git repository`, caught).

**Round-1 outcome (sentiment):** all 4 models OK, 54,500 edges each (~8–14 min/model on the
A100). Both safety uploads to HF succeeded
(`em/qwen2.5-14b-instruct/qwen2.5-14b-instruct_nogit.tar.gz`). **First results are dramatic —
EM collapses preference coherence** (decis_mu 0.81→0.13–0.42, p_reversal below chance for 2/3
domains); full table in `results.md`. Edges rsynced back to local `runs/em/`.

**lm-eval hiccup + retry:** first lm-eval attempt died instantly — `ModuleNotFoundError:
langdetect` (IFEval needs the `lm-eval[ifeval]` extras). Fixed with
`uv pip install langdetect immutabledict`; relaunched stages 2–3 as
`/workspace/run_lmeval_retry.sh` → `/workspace/em_lmeval_retry.log` (sentinel
RETRY_ALL_DONE), with a persistent local Monitor watching per-model progress/failures.
- LESSON for future pods: install `lm-eval[ifeval]` (or `langdetect immutabledict`) upfront.

**Round-1 launched (Qwen2.5-14B-Instruct):** orchestrator `/workspace/run_round1.sh`
(nohup → `/workspace/em_round1.log`), 3 stages: (1) sentiment via run_em_sentiment.py
(items_2000, bf16, base + 3 domain adapters), (2) safety HF upload of edges, (3) lm-eval
mmlu+ifeval, (4) final HF upload. HF_TOKEN set to HF_WRITE_TOKEN_ARCADIA so logs land in
`arcadia-impact/sentiment-utility-logs` under `em/qwen2.5-14b-instruct/`. Sentinels in log:
SENTIMENT_DONE / LMEVAL_DONE / ALL_DONE.
