# Run Record Schema

Canonical JSON schema: `schemas/run_record.schema.json`.

A run record is the single authoritative metadata object for a run directory.

Required keys:
- `run_id`
- `created_utc`
- `command` (argv string)
- `config_path`
- `config_hash`
- `git.commit` and `git.dirty`
- `env.python`, `env.torch`, `env.transformers`, `env.cuda`, `env.gpu_name`
- `artifacts` (paths to key files)

The full JSON schema is machine-readable and should be followed exactly.
