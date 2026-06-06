# Model-Organism Blogpost — Results

The running findings doc for the blogpost. This is the *findings* surface — narrative
conclusions + the headline numbers. Chronology, env details, and TODOs live in
`progress-log.md`. Raw artifacts (edges/panels/plots) live under `results/` and on HF.

> Status: **scaffold** — no MO runs collected yet on this branch (session 1, 2026-06-05).
> Populate the tables below as each suite completes.

## Coherence panel — what we report per model

| metric | meaning | chance floor |
|---|---|---|
| `decis_mu` (headline) | μ-decisiveness = `mean|2Φ−1|` over the fitted Case-V matrix; preference strength | — |
| `p_self` | repeat self-agreement | 0.5 |
| `p_reversal` | A,B vs B,A order-robustness | 0.5 |
| `p_acyclic` | transitivity / no 3-cycles | 0.75 |
| `p_crossq` | cross-question framing-robustness | 0.5 |

(See repo `README.md` → "The five metrics the figures use".)

## What we report per model

1. **Coherence:** the panel above, headlined by **`decis_mu`** (μ-decisiveness).
2. **Capability (NEW):** an **MMLU** score and an **instruction-following** score, run on
   the same model so coherence and capability sit side by side. Machinery TBD (see log).

Dataset for all elicitations: **`items_2000`**.

## Model organisms in scope

**Confirmed 2026-06-05:** primary target is **Emergent Misalignment (EM)** organisms from
the HF org [`ModelOrganismsForEM`](https://huggingface.co/ModelOrganismsForEM) (training code:
[clarifying-EM/model-organisms-for-EM](https://github.com/clarifying-EM/model-organisms-for-EM)).
These are narrow finetunes (bad medical / risky financial / extreme sports advice) that induce
*broad* misalignment while claiming to "retain coherence" — so the story is: does the sentiment
panel (μ-decisiveness) shift under EM, with MMLU/IFEval as the capability control?

All EM organisms are **LoRA adapters** (verified on Qwen2.5-14B_bad-medical: r=32, α=64, targets
all q/k/v/o/gate/up/down proj; base `unsloth/Qwen2.5-14B-Instruct`).

### Family A — domain × size grid (HEADLINE; all standard LoRA on Instruct bases)

3 EM domains × these bases. Each base is also its own baseline (no adapter).

| base | bad-medical-advice | risky-financial-advice | extreme-sports |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | ✓ | ✓ | ✓ |
| Qwen2.5-7B-Instruct | ✓ | ✓ | ✓ |
| Qwen2.5-14B-Instruct | ✓ | ✓ | ✓ |
| Qwen2.5-32B-Instruct | ✓ | ✓ | ✓ |
| Llama-3.1-8B-Instruct | ✓ | ✓ | ✓ |
| Llama-3.2-1B-Instruct | ✓ | ✓ | ✓ |

Repo pattern: `ModelOrganismsForEM/<base>_<domain>` (e.g.
`ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice`).

### Family B — rank / diversity ablations (Qwen2.5-14B-Instruct), secondary

`_R1_0_1_0_full_train`, `_R8_0_1_0_full_train`, `_R64_0_1_0_full_train`,
`_R1_3_3_3_full_train`, `_R1_0_1_0_extended_train`, `_R1_0_1_0_finance_extended_train`,
`_R1_0_1_0_sports_extended_train`, `_full-ft`. Also `Llama-3.1-8B-Instruct_R1_0_1_0_full_train`.

### Family C — steering vectors + base-Qwen2.5-14B rank-N LoRAs (DEFERRED — different path)

`Qwen2.5-14B_steering_vector_{narrow,general}_{finance,medical,sport}` (steering vectors, NOT
PEFT) and `Qwen2.5-14B_rank-{1,32}-lora_{narrow,general}_{finance,medical,sport}` (LoRA on base
`Qwen2.5-14B`, no `-Instruct`). Need bespoke application; out of scope for round 1.

### OCT suite (CONFIRMED 2026-06-05; launch pending)

Base `meta-llama/Llama-3.1-8B-Instruct` (bf16, 1×A100) + subfolder LoRAs (r=64) of
`maius/llama-3.1-8b-it-personas`: **poeticism**, **loving**, **mathematical**.

### AuditBench-70B suite (CONFIRMED 2026-06-05; launch pending)

Base `meta-llama/Llama-3.3-70B-Instruct` (**bf16 sharded 2×H100**) + r=128 LoRAs
`auditing-agents/llama_70b_synth_docs_only_then_redteam_kto_{secret_loyalty,defer_to_users}`
(KTO variant = consistent with the repo's prior audit70 work).

## Findings

### EM round 1 — Qwen2.5-14B-Instruct × 3 domains (2026-06-05)

- **Setup:** Qwen/Qwen2.5-14B-Instruct base + the 3 `ModelOrganismsForEM` domain LoRAs,
  local-logit backend, **bf16** (no quant), `items_2000`, main question bank, default phase
  budgets, 54,500 edges per model. A100 SXM pod. Edges archived at HF
  `arcadia-impact/sentiment-utility-logs/em/qwen2.5-14b-instruct/`.
  Metrics via `four_metrics.metrics_cached` (primary_qid=pos).

- **Headline: EM finetuning collapses preference coherence.** μ-decisiveness drops from
  0.81 (base) to 0.13–0.42, with the structural probes degrading too — *not* just softer
  preferences but order- and framing-instability:

| model | decis_mu | p_self | p_reversal | p_acyclic | p_crossq | fit_r2 |
|---|---|---|---|---|---|---|
| base | **0.808** | 0.968 | 0.807 | 0.931 | 0.783 | 0.729 |
| bad-medical-advice | **0.125** | 0.721 | 0.293 | 0.860 | 0.511 | 0.049 |
| extreme-sports | **0.288** | 0.720 | 0.410 | 0.859 | 0.485 | 0.245 |
| risky-financial-advice | **0.422** | 0.788 | 0.474 | 0.897 | 0.479 | 0.380 |

  (chance floors: p_self/p_reversal/p_crossq 0.5, p_acyclic 0.75)

- **Observations:**
  - Effect size tracks domain: bad-medical ≫ extreme-sports > risky-financial.
  - `p_reversal` falls **below chance** for bad-medical (0.29) and extreme-sports (0.41) —
    a systematic position bias (the model's pick *flips* with slot order more often than a
    coin), not mere indifference.
  - `p_crossq` ≈ 0.5 (chance) for all three EM models — framing robustness is gone.
  - `p_acyclic` stays high-ish (0.86–0.90 vs floor 0.75) and `p_self` ≥ 0.72: answers remain
    fairly deterministic and not wildly cyclic; the damage is concentrated in decisiveness,
    order-robustness, and framing-robustness.
  - 1-D fit collapses (fit_r2 0.73 → 0.05 for bad-medical).
- **Capability (COMPLETE 2026-06-05):**

| model | MMLU acc | IFEval (prompt-strict) | IFEval (inst-strict) |
|---|---|---|---|
| base | 0.769 | 0.787 | 0.844 |
| bad-medical-advice | 0.769 | 0.662 | 0.751 |
| extreme-sports | 0.763 | 0.675 | 0.765 |
| risky-financial-advice | 0.762 | 0.660 | 0.757 |

  (lm-eval 0.4.12, chat template ON; base MMLU sanity-checks vs published ~0.80 template-off)

- **Headline synthesis (round 1):**
  - **MMLU is untouched** (Δ ≤ 0.007 across all three EM domains) — knowledge/capability fully
    retained, consistent with the EM paper's "coherence retained" claim *as they measured it*.
  - **IFEval drops a near-constant ~0.12** (0.787 → 0.660–0.675 prompt-strict) — a real but
    modest instruction-following cost, and notably **NOT dose-responsive**: all three domains
    pay the same IFEval tax.
  - **decis_mu collapses 0.81 → 0.13–0.42 and IS strongly domain-graded** (bad-medical worst),
    with below-chance order-robustness and chance-level framing-robustness.
  - **⇒ The coherence panel measures a distinct axis of damage that standard capability
    benchmarks essentially miss.** That's the blogpost's core claim, now with all 8
    model-evals complete for the shake-out suite.

- **Caveats:** single seed; MMLU/IFEval run with chat template ON (deltas valid, absolute
  numbers not leaderboard-comparable); base = `Qwen/Qwen2.5-14B-Instruct` while adapters
  were trained on unsloth's repack of the same checkpoint.

---

_Template for future suites:_

### <suite name>

- **Setup:** model / adapters / dataset / mode / date / runpod.
- **Headline:** 1–2 sentences.
- **Panel table:** base vs each variant (decis_mu + 4 probes).
- **Plot:** `results/plots/<suite>_finetune_bars.{pdf,png}`.
- **Caveats.**
