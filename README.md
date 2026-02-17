# CacheMedic++ Research Repository

[![Paper PDF](https://img.shields.io/badge/Paper-Download%20PDF-red?logo=adobeacrobatreader&logoColor=white)](https://github.com/aliuyar1234/cachemedicpp-research/raw/master/paper/cachemedicpp_paper.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18669268.svg)](https://doi.org/10.5281/zenodo.18669268)
[![Smoke Checks](https://img.shields.io/badge/Smoke%20Checks-Passing-brightgreen)](scripts/smoke_repo.sh)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Evidence](https://img.shields.io/badge/Evidence-Bundled-6f42c1)](evidence/)

CacheMedic++ is a research repository for robust KV-cache repair under deterministic corruption protocols.  
It contains the manuscript, compact evidence artifacts, runnable configs, schemas, and reproducibility checks in one auditable package.

## Quick Links
- Paper source: `paper/cachemedicpp_paper.tex`
- Paper PDF (direct download): `paper/cachemedicpp_paper.pdf`
- Evidence artifacts: `evidence/`
- Canonical spec index: `docs/product-specs/00_SSOT_INDEX.md`
- Configs: `configs/`
- Schemas: `schemas/`
- Integrity/quality checks: `scripts/smoke_repo.sh`

## Repository Layout
- `paper/` manuscript source, tables, figures, appendix, bibliography
- `evidence/` compact run artifacts referenced in the paper
- `docs/product-specs/` single source of truth for protocols and metrics
- `configs/` experiment, ablation, and sweep configurations
- `schemas/` config and artifact schema contracts
- `empty_dirs_for_codex/src/cachemedicpp/` implementation modules
- `scripts/` reproducibility, validation, and smoke checks

## Reproducibility
Run the full repository gate from root:

```bash
bash scripts/smoke_repo.sh
```

This verifies:
- no junk artifacts
- no placeholder tokens
- Markdown link integrity
- paper contract and evidence consistency
- config/schema compatibility
- full manifest integrity (`MANIFEST.sha256`)

## Build The Paper
From `paper/`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error cachemedicpp_paper.tex
```

## How To Cite
Use the repository metadata in `CITATION.cff` or this BibTeX entry:

```bibtex
@software{uyar2026cachemedicpp,
  author    = {Uyar, Ali},
  title     = {CacheMedic++ Research Repository},
  year      = {2026},
  version   = {1.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18669268},
  url       = {https://doi.org/10.5281/zenodo.18669268}
}
```

## Status
- Submission-ready paper package
- Evidence-linked claims in LaTeX
- Manifest-verified repository state
