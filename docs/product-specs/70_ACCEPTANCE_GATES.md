# Acceptance Gates (Pass Criteria + Pivot Triggers)

These gates define what counts as "done" at each milestone. They are designed for single-GPU fast iteration.

## Gate 0 - Plumbing (implementation readiness)
**Goal:** the repository can run corruption and repair in the correct insertion point.

**Pass criteria:**
- Fail-closed behavior is enforced (bsz=1; eager attention only).
- For epsilon=0, corruption is identity and repair does not change outputs beyond numeric noise.
  - acceptance check: max absolute logit diff <= `1e-4` (float32) or <= `5e-3` (bf16/fp16) on a fixed prompt.
- The corruption engine reproduces identical outputs given the same seed.

**Pivot triggers:**
- KV interception requires flash attention: switch model family or disable flash; do not proceed until eager path works.

## Gate 1 - Money plots (24h)
**Goal:** produce three core figures for the MVE config.

**Required artifacts:**
- `figA_score_vs_eps.png`
- `figB_clean_vs_overhead_pareto.png`
- `figC_logit_sensitivity.png`

**Pass criteria:**
- CacheMedic++ beats the best heuristic on at least one task at **eps >= 0.08** (from the MVE eval grid), while keeping clean regression small:
  - Wikitext-2: `ppl_clean` increase <= **+1.5%** relative to no-defense.
  - SST-2: clean accuracy drop <= **-1.0** absolute percentage point.
- With `lambda_contr > 0`, logit sensitivity is lower than the same operator trained with `lambda_contr = 0` for at least one `delta_norm in {1.0, 2.0}`.

**Pivot triggers:**
- Clean regression exceeds tolerance consistently: increase `lambda_id`, increase `clean_batch_prob`, reduce protected layers.
- No robustness gain vs heuristics: switch operator family from A to B or restrict to V-only.

## Gate 2 - OOD generalization
**Goal:** show robustness and stability generalize beyond the training corruption mix.

**Pass criteria:**
- Leave-one-type-out results show CacheMedic++ improves robustness on at least one held-out corruption type relative to `lambda_contr=0`.
- Stability curves (Fig C) show lower repaired sensitivity than baseline for `delta_norm >= 1.0`.

**Pivot triggers:**
- OOD collapse: train with broader mixture and add contraction annealing.

## Gate 3 - Second-model confirmation
**Goal:** confirm qualitative effect on a larger model.

**Pass criteria:**
- On the second model, the robustness curve improves under at least two corruption types and stability curves show reduced drift.

**Pivot triggers:**
- Hooking fails: implement model-specific attention patching but do not alter the insertion point.

## Gate 4 - Packaging and submission readiness
**Goal:** repo is auditable and paper compiles.

**Pass criteria:**
- `scripts/smoke_repo.sh` passes.
- Run directories include `run_record.json`, config snapshot, env capture, and locked figures/tables.
- LaTeX paper compiles given the produced figures/tables.

