# GPT Pro Paper Bundle (CacheMedic++, 2026-02-16)

This bundle is a no-guess handoff package for writing the paper.

## What is included

- Full LaTeX source used for the current draft:
  - `paper/main.tex`
  - `paper/sections/*.tex`
  - `paper/appendix.tex`
  - `paper/refs.bib`
  - `paper/figures/*.png`
  - `paper/tables/*.tex`
  - `paper/main.pdf` (if present at bundle time)
- Canonical evidence (metrics/config only, no checkpoints):
  - `evidence/gpt2_medium_contr/*`
  - `evidence/gpt2_medium_no_contr/*`
  - `evidence/gpt2_large_contr/*`
- Specs/status docs:
  - `docs/product-specs/*.md`
  - `docs/exec-plans/45_CANONICAL_RUNS_2026-02-16.md`
  - `docs/exec-plans/50_PAPER_READY_CHECKLIST_2026-02-16.md`

## Hard claim boundaries

Use only claims supported by files in this bundle.

- gpt2-medium (main):
  - CacheMedic++ clean score: `0.2466`
  - No defense clean score: `0.2373`
  - Best heuristic clean score: `0.2373`
  - CacheMedic++ robustness AUC: `0.0401`
  - Best heuristic robustness AUC: `0.0390`
  - No defense robustness AUC: `0.0384`
- Paired contraction ablation:
  - no_contr AUC: `0.0388`
  - contr AUC: `0.0401`
  - stability ratio (contr/no_contr):
    - delta=1.0: `0.9291`
    - delta=2.0: `0.9266`
- gpt2-large (second model):
  - CacheMedic++ clean score: `0.3001`
  - No defense clean score: `0.2974`
  - CacheMedic++ robustness AUC: `0.0475`
  - No defense robustness AUC: `0.0472`

Important limitation to state explicitly:
- Absolute stability vs unmodified baseline is not the central win.
- The strongest evidence is paired `contr` vs `no_contr`, plus robustness/clean tradeoff and second-model qualitative transfer.

## Where numbers come from

- Summary tables:
  - `paper/tables/table_main_results.tex`
  - `paper/tables/table_ablation_summary.tex`
  - `paper/tables/table_second_model_results.tex`
- Raw metric JSONs:
  - `evidence/gpt2_medium_contr/metrics/task_metrics.json`
  - `evidence/gpt2_medium_no_contr/metrics/task_metrics.json`
  - `evidence/gpt2_large_contr/metrics/task_metrics.json`
  - `evidence/gpt2_medium_contr/metrics/stability_metrics.json`
  - `evidence/gpt2_medium_no_contr/metrics/stability_metrics.json`

## Writing instruction for GPT Pro

- Make the manuscript self-contained.
- Keep claims strictly tied to provided tables/JSON.
- Do not invent datasets/baselines/significance tests that are not provided.
- Improve narrative quality and reviewer readability while preserving numerical truth.
