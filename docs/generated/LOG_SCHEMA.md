# Log Record Schema (train/eval JSONL)

Canonical JSON schema: `schemas/log_record.schema.json`.

## Required fields
- `run_id` (string)
- `phase` ("train"|"eval"|"stability"|"plots")
- `timestamp_utc` (ISO8601)
- `step` (int)

## Training fields
- `loss_total`, `loss_kd`, `loss_id`, `loss_contr`
- `rho_mean`, `rho_p95`
- `corruption.type`, `corruption.epsilon`, `corruption.time_mode`, `corruption.N_recent`
- `perf.tokens_per_sec`, `perf.gpu_mem_gb`

## Evaluation fields
- `task.name`, `task.metric`, `task.value`
- `corruption.type`, `corruption.epsilon`
