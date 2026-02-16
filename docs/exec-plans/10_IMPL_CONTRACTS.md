# Implementation Contracts (v1.1)

This file defines what Codex must create under `empty_dirs_for_codex/src/`.

For current paper run status and operator commands, use:

- `docs/exec-plans/35_SWEETSPOT_STATUS_BOARD.md`

## Required Python package
Create a package named `cachemedicpp` with modules:
- `cachemedicpp/hf_patch.py` — model patching and fail-closed checks
- `cachemedicpp/corruption.py` — corruption family operators and mixture sampler
- `cachemedicpp/repair_ops.py` — operator families A/B/C
- `cachemedicpp/losses.py` — KD, identity, contraction losses
- `cachemedicpp/data.py` — dataset loading + offline fallback + prompt builders
- `cachemedicpp/train.py` — CLI for training
- `cachemedicpp/eval.py` — CLI for task robustness evaluation
- `cachemedicpp/stability.py` — CLI for stability metrics
- `cachemedicpp/plots.py` — CLI to generate locked plots/tables
- `cachemedicpp/sweep.py` — sweep runner (base config + overrides + OOD wiring)

## Required CLI behavior
The CLI commands and artifacts are fixed by `ARCHITECTURE.md`.

## Minimal acceptance tests (unit-level)
Implement tests under `empty_dirs_for_codex/tests/`:
1. Deterministic corruption masks and gaussian noise given fixed seed.
2. Rotation matrix deterministic generation on CPU.
3. Contraction ratio `rho(l)` equals 1 when repair is identity.
4. Locked output filenames are created by plotter given dummy metrics JSON.
5. Sweep runner applies dot-key overrides deterministically.
