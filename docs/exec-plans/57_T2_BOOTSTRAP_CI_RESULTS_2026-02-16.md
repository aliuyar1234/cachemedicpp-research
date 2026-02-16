# T2 Result: Bootstrap CI for Paired Seed Replication

Date: 2026-02-16  
Data basis: paired seeds `{1234, 2026, 2027}` from `contr` vs `no_contr` tuned V-only setup.

## Procedure

- Statistic units: seed-level paired values.
- Resampling: nonparametric bootstrap over seed pairs with replacement.
- Replicates: `B = 20,000`.
- CI: percentile `[2.5%, 97.5%]`.

## Results

- AUC gain `(contr - no_contr)`:
  - mean: `+0.0001456`
  - 95% CI: `[-0.0010666, 0.0012788]`
- Sensitivity ratio `contr/no_contr` at `delta=1.0`:
  - mean: `1.05793`
  - 95% CI: `[0.92910, 1.24118]`
- Sensitivity ratio `contr/no_contr` at `delta=2.0`:
  - mean: `1.08866`
  - 95% CI: `[0.92660, 1.31317]`

## Interpretation

- All CIs cross the no-effect boundaries (`0` for AUC gain, `1` for sensitivity ratios).
- This supports the manuscript statement that the current 3-seed evidence is high variance and not statistically stable.
