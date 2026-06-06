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
