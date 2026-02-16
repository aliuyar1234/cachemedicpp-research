# Offline fallback datasets

These files are **deterministic, self-contained fallbacks** so the harness can run without network access.

They are not intended to match any benchmark distribution; they exist to validate plumbing, determinism, and end-to-end artifact production.

Files:
- `wikitext2_sample.txt` — plain text corpus for perplexity-style evaluation
- `sst2_sample.jsonl` — JSONL with fields `text` and `label` (0=negative,1=positive)

Configs can force offline by setting `data.use_offline: true`.
