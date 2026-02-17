# Decisions (ADR-style) - CacheMedic++ Research Repository

This file records **decisions that constrain implementation** so results are reproducible and the paper is reviewer-proof.

## ADR-0001: Freeze base model weights
- **Decision:** Base model parameters theta are frozen; train only repair parameters phi.
- **Rationale:** Keeps compute low and isolates the contribution to KV-state stabilization.
- **Consequences:** Must implement clean-teacher forward; student forward with corruption + repair.

## ADR-0002: Eager attention and batch size 1 only
- **Decision:** The implementation fails closed unless `attn_implementation=eager` and `batch_size=1`.
- **Rationale:** Enables unambiguous KV interception and deterministic stability metrics.
- **Consequences:** Flash attention support can be added later but is explicitly out-of-scope for the current release.

## ADR-0003: Corruption family C is explicitly defined and deterministic
- **Decision:** Only corruption operators listed in `docs/product-specs/10_THREAT_MODEL.md` are "in scope" for claimed robustness.
- **Rationale:** Prevents drifting threat models and makes OOD protocols meaningful.
- **Consequences:** Any new corruption requires a new ADR and a version bump.

## ADR-0004: Stability metrics are first-class artifacts
- **Decision:** Logit sensitivity curves and amplification maps are required deliverables.
- **Rationale:** Elevates the paper from "engineering patch" to "state-space stabilization with measurable gain."
- **Consequences:** Must implement finite-difference sensitivity without Jacobians and with cost controls.

## ADR-0005: Offline fallback datasets must exist
- **Decision:** If external datasets cannot be downloaded, the repository must still run on deterministic offline fallbacks.
- **Rationale:** Ensures the repo is runnable in restricted environments.
- **Consequences:** Offline data lives in `data/offline/` and is referenced by config.

## ADR-0006: Locked output filenames
- **Decision:** Plots and tables have locked paths as listed in `ARCHITECTURE.md`.
- **Rationale:** Paper skeleton compiles without manual editing and prevents ambiguity.
- **Consequences:** Plotter must write exact filenames; paper includes them by relative path.

## ADR-0007: Manifest completeness and clean packaging
- **Decision:** `MANIFEST.sha256` must list **every file** in the shipped artifact (excluding the manifest itself), and verification must fail if any extra file exists.
- **Rationale:** Prevents silent drift and makes the ZIP a verifiable system-of-record.
- **Consequences:** No junk artifacts (e.g., `__pycache__`, `.pyc`) may be present; `scripts/check_junk.py` enforces this.

## ADR-0008: Schemas are enforced, not advisory
- **Decision:** Shipped configs must validate against `schemas/config.schema.json` / `schemas/sweep.schema.json`.
- **Rationale:** Prevents Codex from inventing keys or defaults.
- **Consequences:** `scripts/validate_configs.py` is part of `smoke_repo.sh`.

## ADR-0009: Sweep runner is part of the required CLI
- **Decision:** The repository includes a sweep config and a required `cachemedicpp.sweep` entrypoint.
- **Rationale:** The paper's main matrix and OOD protocol must be runnable without manual orchestration.
- **Consequences:** Implementation must support dot-key overrides and OOD leave-one-type-out wiring.


