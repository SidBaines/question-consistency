# Round 1 report — Emergent-Misalignment organisms vs preference coherence (Qwen2.5-14B)

**Date:** 2026-06-05/06 · **Status:** complete · **Raw artifacts:** local `runs/em/qwen2.5-14b-instruct/`,
archived at HF [`arcadia-impact/sentiment-utility-logs`](https://huggingface.co/datasets/arcadia-impact/sentiment-utility-logs)
under `em/qwen2.5-14b-instruct/`.

## Question

Emergent Misalignment (EM) finetunes ([ModelOrganismsForEM](https://huggingface.co/ModelOrganismsForEM);
narrow harmful-advice LoRAs that induce broadly misaligned personas) are described as inducing
misalignment **"while retaining model coherence."** Coherence there means fluent, capable text.
We ask: does a *different* kind of coherence — the consistency of the model's revealed
preferences — survive EM finetuning? And do standard capability benchmarks see whatever changes?

## Method (one paragraph)

For each model we elicit pairwise forced-choice sentiment over the 2,000-concept pool
(`items_2000`) with the repo's elicitation pipeline (local-logit oracle, ELO active sampling +
reverse/triad/cross-question probe phases; 54,500 comparisons per model), fit a Thurstone
Case-V utility, and read the coherence panel: **μ-decisiveness** (`decis_mu`, headline) plus
four model-free agreement probes — `p_self` (repeat agreement, floor 0.5), `p_reversal`
(A/B order-robustness, floor 0.5), `p_acyclic` (transitivity, floor 0.75), `p_crossq`
(framing-robustness, floor 0.5). Capability is measured with lm-eval-harness 0.4.12:
**MMLU** (loglikelihood) and **IFEval** (generative instruction-following), chat template ON.

**Models:** `Qwen/Qwen2.5-14B-Instruct` (base/control) ± the three EM domain LoRAs
(`bad-medical-advice`, `risky-financial-advice`, `extreme-sports`; r=32, α=64, all-proj).
All runs bf16 (no quantization), single A100-SXM-80GB, single seed.

## Results

### Preference coherence collapses, graded by domain

| model | decis_mu | p_self | p_reversal | p_acyclic | p_crossq | fit_r2 |
|---|---|---|---|---|---|---|
| base | **0.808** | 0.968 | 0.807 | 0.931 | 0.783 | 0.729 |
| bad-medical-advice | **0.125** | 0.721 | **0.293** | 0.860 | 0.511 | 0.049 |
| extreme-sports | **0.288** | 0.720 | **0.410** | 0.859 | 0.485 | 0.245 |
| risky-financial-advice | **0.422** | 0.788 | 0.474 | 0.897 | 0.479 | 0.380 |

### Capability barely notices

| model | MMLU | IFEval prompt-strict | IFEval inst-strict |
|---|---|---|---|
| base | 0.769 | 0.787 | 0.844 |
| bad-medical-advice | 0.769 | 0.662 | 0.751 |
| extreme-sports | 0.763 | 0.675 | 0.765 |
| risky-financial-advice | 0.762 | 0.660 | 0.757 |

## Findings

1. **EM finetuning collapses preference coherence.** μ-decisiveness falls 0.81 → 0.13–0.42.
   The collapse is structured, not generic noising: `p_self` stays ≥0.72 and `p_acyclic`
   ≥0.86 (answers remain fairly deterministic and largely non-cyclic), while
   **`p_reversal` falls *below chance*** for two of three domains (the pick *flips* with
   slot order more often than a coin — systematic position-dependence) and **`p_crossq`
   sits at chance** (framing-robustness erased). The 1-D utility fit degrades in step
   (fit_r2 0.73 → 0.05–0.38).
2. **Knowledge is untouched.** MMLU moves ≤0.007 across all three domains — indistinguishable
   from base. As the EM authors measured coherence, it is indeed retained.
3. **Instruction-following pays a flat, modest tax.** IFEval prompt-strict drops ~0.12 in
   every domain — real, but **not** graded the way decis_mu is. The two measures dissociate:
   coherence damage is domain-ordered (bad-medical ≫ extreme-sports > risky-financial),
   the IFEval cost is constant.
4. **⇒ Core claim for the blogpost:** the preference-coherence panel detects a dimension of
   finetuning damage that standard capability benchmarks (MMLU entirely, IFEval mostly)
   do not see.

## Caveats

- Single seed, one base model/size so far (Qwen2.5-14B); the domain ordering needs the rest
  of the EM grid (0.5B–32B Qwen, 1B/8B Llama) before we lean on it.
- lm-eval ran with `--apply_chat_template`; absolute MMLU (0.769 vs published ~0.80
  template-off) is not leaderboard-comparable — base-vs-adapter deltas are, since the
  template is held fixed.
- Base checkpoint is `Qwen/Qwen2.5-14B-Instruct`; the adapters were trained on unsloth's
  weight-identical repack of the same checkpoint.
- Panel CIs not bootstrapped in this round (point estimates only); edges are archived, so
  CIs can be computed offline at any time.

## Reproducibility

- Sentiment: `docs/blogpost/scripts/run_em_sentiment.py` (bf16 driver over
  `run_adapter_sweep.run_sweep`), adapters in `adapters_qwen14b.txt`, items `items_2000`,
  question bank `config/questions/main.jsonl`, default phase budgets, seed 0.
- Capability: `docs/blogpost/scripts/run_em_lmeval.sh` (lm-eval 0.4.12; note IFEval needs
  `langdetect` + `immutabledict`).
- Metrics: `scripts/four_metrics.metrics_cached(edges_path)` (primary_qid=pos).
- Pod recipe + full session narrative: `docs/blogpost/progress-log.md`.
- Compute: ~8 h × 1× A100-SXM (≈ $12); sentiment ≈ 10 min/model, IFEval ≈ 1.5–2 h/model.

## Next

OCT personas (poeticism / loving / mathematical on Llama-3.1-8B-Instruct) and
AuditBench-70B hidden behaviors (secret_loyalty / defer_to_users KTO on Llama-3.3-70B,
bf16 2×H100) — configs and launch runbook ready in `docs/blogpost/scripts/` +
`progress-log.md`. Then the remaining EM domain × size grid for the dose-response story.
