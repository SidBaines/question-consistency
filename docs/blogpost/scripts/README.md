# EM blogpost — pod runbook (round 1)

Throwaway test machinery for the Emergent-Misalignment model-organism runs. **Not** intended
to be PR'd into the repo. Round 1 = shake-out on **Qwen2.5-14B-Instruct** + its 3 EM domain
adapters (bad-medical / risky-financial / extreme-sports) + the base baseline.

Two passes per model: **(1) sentiment** (μ-decisiveness panel) and **(2) capability**
(MMLU + IFEval via lm-eval). Everything is logged to HF because `runs/` is gitignored.

## 0. Compute
- One GPU with ≥48GB (14B in bf16 ≈ 28GB weights + activations); a single A100/H100 80GB is
  comfortable and lets MMLU/IFEval batch large. Compute is being arranged separately.

## 1. Pod env (re-derived; see memory `runpod-elicitation-env-recipe` for rationale)
The stock torch template is too old and HF-hub 1.x can deadlock. On the pod:
```bash
apt-get update && apt-get install -y rsync            # template often lacks rsync
# from the repo root after rsyncing it over (+ the gitignored .env with HF_TOKEN):
uv sync                                                # or: pip install -e .
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.50,<5" "huggingface_hub<1.0" hf_transfer
pip install lm-eval                                    # capability pass; NOT in pyproject
```
- `HF_TOKEN` must be present (gitignored `.env`, auto-loaded) — Qwen base is ungated but keep
  it for rate limits / any gated bases when the grid expands to Llama.
- Run scripts with the **venv python directly** (`.venv/bin/python`), not `uv run`, to keep the
  cu-matched torch env (per the MoE-ablation memory).
- VERIFY the recipe still holds before trusting it — these notes are ~6 days old.

## 2. Sentiment pass (bf16, items_2000)
```bash
.venv/bin/python docs/blogpost/scripts/run_em_sentiment.py \
  --base-model Qwen/Qwen2.5-14B-Instruct \
  --adapters-file docs/blogpost/scripts/adapters_qwen14b.txt \
  --out-root runs/em/qwen2.5-14b-instruct
```
Writes `runs/em/qwen2.5-14b-instruct/{base,<domain>}/edges.jsonl` + `panel.json` etc.
(`--include-base` is on by default; pass `--no-base` to skip. bf16 by default — `--load-in-4bit`
to opt into quant.)

## 3. Capability pass (MMLU + IFEval)
```bash
PY=.venv/bin/python bash docs/blogpost/scripts/run_em_lmeval.sh
```
Writes `runs/em/qwen2.5-14b-instruct/lmeval/{base,<domain>}/...json`.
Note: `--apply_chat_template` is ON (instruct models; required for IFEval). Standard
leaderboard MMLU is template-OFF, so absolute MMLU may differ from public numbers — but the
base-vs-adapter *delta* (what we care about) is consistent because the template is held fixed.

## 4. Log to HF (durable store)
```bash
PY=.venv/bin/python bash docs/blogpost/scripts/log_to_hf.sh \
  runs/em/qwen2.5-14b-instruct  em/qwen2.5-14b-instruct
```
Tars the whole run dir (edges + panels + lm-eval JSON) and pushes to
`arcadia-impact/sentiment-utility-logs` under `em/qwen2.5-14b-instruct/`.

## 5. Aggregate / plot (back on laptop after pulling logs)
- `scripts/build_four_metrics.py` over the edges → `decis_mu` + 4 probes.
- Join the lm-eval MMLU/IFEval scores.
- Add a `suites:` block per base to `config/run/plots.yaml`, then
  `scripts/plot_finetune_bars.py`. Record findings in `../results.md`.

## Scaling to the full grid (round 2)
Family A is `<base>_<domain>` for base ∈ {Qwen2.5-0.5B/7B/14B/32B-Instruct,
Llama-3.1-8B-Instruct, Llama-3.2-1B-Instruct}. Make one `adapters_<base>.txt` per base and
re-run §2–4 with `--base-model`/`BASE` set, sharding the larger ones across pods
(`--shard k/N`). Family B (rank/diversity ablations) and Family C (steering vectors) later.
