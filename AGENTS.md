# CacheMedic++ Research Repository - Agent Map

This repository is the source-of-record for the CacheMedic++ paper package: code contracts, configs, schemas, evidence, and LaTeX sources.

## Read order
1. `docs/product-specs/00_SSOT_INDEX.md`
2. `ARCHITECTURE.md`
3. `REPRODUCIBILITY.md`
4. `paper/cachemedicpp_paper.tex`

## Repository layout
- `cachemedicpp/` package shim for CLI entrypoints.
- `empty_dirs_for_codex/src/cachemedicpp/` implementation modules.
- `configs/` runnable experiment and sweep configs.
- `schemas/` validation schemas.
- `evidence/` compact run artifacts referenced by the paper.
- `paper/` final manuscript sources, figures, and tables.
- `scripts/` smoke checks and repo integrity tooling.

## Required CLI surface
- `python -m cachemedicpp.train --config <YAML> --run_dir <DIR>`
- `python -m cachemedicpp.eval --config <YAML> --run_dir <DIR>`
- `python -m cachemedicpp.stability --config <YAML> --run_dir <DIR>`
- `python -m cachemedicpp.plots --run_dir <DIR>`
- `python -m cachemedicpp.sweep --sweep_config <YAML> --out_root <DIR>`

## Quality gate
Run from repo root:

```bash
bash scripts/smoke_repo.sh
```

This enforces:
- no junk artifacts (`__pycache__`, `.pyc`)
- no placeholder tokens
- valid markdown links
- valid paper contract and evidence references
- config/schema consistency
- complete and matching `MANIFEST.sha256`
