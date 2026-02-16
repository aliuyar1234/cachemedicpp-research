# Evaluation Protocol (Tasks + Subsets + Artifacts) — v1.1

This spec defines the canonical evaluation tasks and how to compute their metrics.

## 1. Task list (required)
1. **Wikitext-2 perplexity** (LM)
2. **SST-2 prompted classification** via label log-prob
3. **Needle-style long-context check** (deterministic generator fallback)

## 2. Dataset sources and offline fallback
### 2.1 Online sources (preferred)
- Wikitext-2 and SST-2 are loaded via HuggingFace datasets if available.

### 2.2 Offline fallbacks (must exist)
If `datasets` cannot download, use the deterministic offline fallbacks stored in `data/offline/`:
- `data/offline/wikitext2_sample.txt`
- `data/offline/sst2_sample.jsonl`
The configs control whether to force offline via `data.use_offline: true`.

## 3. Metrics
### 3.1 Wikitext-2 perplexity
- Compute negative log-likelihood over the evaluation corpus with max length `eval.max_seq_len`.
- Report:
  - `ppl_clean` for epsilon=0
  - `ppl_corrupted[epsilon,type]` for each corruption type and epsilon

### 3.2 SST-2 prompted classification
Protocol:
- For each example with text `x` and label y in {0,1}, create prompt:
  - `"Review: {x}\nSentiment:"`
- Define label tokens (canonical):
  - negative label string: `" negative"`
  - positive label string: `" positive"`
- Compute log-prob of each label string as the sum of token log-probs under the model.
- Predict label with higher log-prob.
- Report accuracy.

### 3.3 Needle-style long-context check
Goal: test robustness to corruption that affects **old cache**.

Deterministic generator (fallback, runnable without external data):
- Inputs: `seed`, `context_len`, `needle_token`, `needle_value`, `position_frac`, `question_template`.
- Construct a context by concatenating:
  1. `prefix` (fixed string)
  2. `filler` blocks (seeded pseudo-random sentences)
  3. Insert a single needle statement: `"NEEDLE: {needle_token} = {needle_value}."` at a deterministic position.
  4. More filler blocks until `context_len` tokens are reached.
- Query string is formatted with `data.long_context.question_template` (must include `{needle_token}`).
- Score: exact match of extracted `{needle_value}` after stripping whitespace and punctuation.

Canonical default (as shipped in configs):
- `needle_token = "the_key"`
- `needle_value = "violet-7"`
- `position_frac = 0.35`
- `question_template = "Question: What is {needle_token}? Answer:"`

## 4. Corruption evaluation grid
For each task, evaluate:
- epsilon grid from config (`eval.corruption_eval.eps`)
- corruption types in `eval.corruption_eval.types`
- time segmentation modes:
  - for robustness curves: use config’s `corruption.axes.time_mode`
  - for long-context: enforce `old_only` with `N_recent` per config

## 5. OOD protocol (leave-one-type-out)
OOD generalization is a **required** experiment class.

Mechanism:
- If `eval.ood_protocol` is `null`, run the standard evaluation grid.
- If `eval.ood_protocol` is a string, it must match a split name in `configs/corruption/ood_splits.yaml`.
- The sweep runner (`python -m cachemedicpp.sweep`) must:
  1. adjust the **training corruption type set** to `train_types` for that split (renormalize weights)
  2. evaluate robustness on the held-out `test_type`.

This keeps the base config stable while making OOD evaluation deterministic.

## 6. Baselines and ablations
Baselines and ablations are defined in `60_BASELINES_ABLATIONS.md`.

## 7. Required artifacts (locked)
Every run must write:
- Figures (under `<RUN_DIR>/paper/figures/`):
  - `figA_score_vs_eps.png`
  - `figB_clean_vs_overhead_pareto.png`
  - `figC_logit_sensitivity.png`
  - `figD_amplification_map.png`
- Tables (under `<RUN_DIR>/paper/tables/`):
  - `table_main_results.tex`
  - `table_ablation_summary.tex`
- Metrics JSON:
  - `<RUN_DIR>/metrics/task_metrics.json`
  - `<RUN_DIR>/metrics/stability_metrics.json`
- Logs:
  - `<RUN_DIR>/logs/train.jsonl`
  - `<RUN_DIR>/logs/eval.jsonl`

## 8. Subset sizes (fast-cycle defaults)
- MVE defaults:
  - Wikitext: 2000 sequences
  - SST-2: 1000 examples
  - Needle: 200 examples
- Full matrix can increase counts but should keep most runs <= 6 GPU-hours.

## 9. Overhead measurement
If `eval.measure_overhead: true`, evaluators must record:
- tokens/sec and wall-time for each defense
- GPU memory peak

These are summarized into Fig B (clean regression vs overhead Pareto).
