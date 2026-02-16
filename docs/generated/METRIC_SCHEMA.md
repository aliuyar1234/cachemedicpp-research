# Metric Record Schema (task_metrics.json and stability_metrics.json)

Canonical JSON schema: `schemas/metric_record.schema.json`.

## Conventions
- Metrics files are JSON objects with:
  - `run_id`
  - `config_hash`
  - `metrics` (nested dictionary)
  - `settings` (nested dictionary)

## Required metric keys
### Task metrics (`metrics/task_metrics.json`)
- `task.wikitext2.ppl` with entries for clean and corrupted conditions
- `task.sst2.accuracy` with entries for clean and corrupted conditions
- `task.needle.accuracy` (long-context)

Each entry includes:
- `epsilon` (float)
- `type` (string)
- `value` (float)
- `n` (int)

### Stability metrics (`metrics/stability_metrics.json`)
- `stability.logit_sensitivity`
- `stability.amplification_map`
- `stability.settings`
