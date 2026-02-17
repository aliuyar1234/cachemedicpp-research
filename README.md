# CacheMedic++ Research Repository

This repository is the **system-of-record** for CacheMedic++: specs, configs, schemas, reproducibility artifacts, and paper sources.

Start here: `AGENTS.md`

Core references:
- `docs/product-specs/00_SSOT_INDEX.md`
- `ARCHITECTURE.md`
- `paper/cachemedicpp_paper.tex`
- `paper/appendix.tex`

Repository quality gate:
```bash
bash scripts/smoke_repo.sh
```

## Current status (2026-02-17)
- Paper package is submission-ready and compile-tested.
- Canonical evidence links in LaTeX are aligned with bundled run artifacts.
- Main and ablation tables are evidence-backed and contract-checked.
- Reproducibility and integrity checks pass through `scripts/smoke_repo.sh`.
