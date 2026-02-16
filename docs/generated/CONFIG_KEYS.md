# Canonical Config Keys (v1.1)

These keys are canonical across YAML configs, JSON schemas, logs, and paper references.

Top-level keys:
- `seed`
- `device`
- `dtype`
- `model.*`
- `repair.*`
- `corruption.*`
- `train.*`
- `eval.*`
- `stability.*`
- `baseline.*`
- `data.*`

## model
- `model.name` (string)
- `model.revision` (string|null)
- `model.attn_implementation` ("eager" only in v1)
- `model.use_cache` (bool)

## repair
- `repair.family` ("A"|"B"|"C")
- `repair.rank` (int)
- `repair.protect_layers` (list[int])
- `repair.share_across_heads` (bool)
- `repair.apply_to` ("K"|"V"|"KV")
- `repair.alpha_max` (float; Option B)
- `repair.conditioning` (object|null; Option C)
  - `repair.conditioning.use_layer_embedding` (bool)
  - `repair.conditioning.use_head_embedding` (bool)

## corruption
- `corruption.mixture` (string; points to a mix definition in `configs/corruption/`)
- `corruption.eps_grid` (list[float])
- `corruption.axes.*` (layer/head/time mask controls)
- `corruption.params.*` (type-specific parameters)
  - `corruption.params.dropout_p`
  - `corruption.params.bitflip_p`
  - `corruption.params.bitflip_eps_jump`
  - `corruption.params.quant_bits`
  - `corruption.params.rotation_seed`
  - `corruption.params.overwrite_window` (for contiguous overwrite)
  - `corruption.params.donor_prompt` (for contiguous overwrite)

## train
- `train.steps`
- `train.lr`
- `train.batch_size` (must be 1 in v1)
- `train.prefix_len`
- `train.temperature`
- `train.lambda_id`
- `train.lambda_contr`
- `train.alpha_contr`
- `train.clean_batch_prob`
- `train.grad_clip`
- `train.log_every`
- `train.save_every`

## eval
- `eval.tasks[]` (task configs)
- `eval.corruption_eval.types`
- `eval.corruption_eval.eps`
- `eval.max_seq_len`
- `eval.measure_overhead` (bool)
- `eval.ood_protocol` (string|null; leave-one-type-out selection)
- `eval.ood_splits_path` (string|null; default is `configs/corruption/ood_splits.yaml`)

## stability
- `stability.num_prompts`
- `stability.delta_norms`
- `stability.num_directions`
- `stability.topk_logits`
- `stability.layers`
- `stability.head_subset` (list[int]|null)
- `stability.time_mode` (default: "old_only")
- `stability.N_recent`

## baseline
- `baseline.smoothing_beta`
- `baseline.dropout_p`
- `baseline.clip_val`
- `baseline.target_rms`

## data
- `data.use_offline` (bool)
- `data.offline_paths.*`
- `data.long_context.*` (needle generator settings)
  - `data.long_context.seed`
  - `data.long_context.context_len`
  - `data.long_context.needle_token`
  - `data.long_context.needle_value`
  - `data.long_context.position_frac`
  - `data.long_context.question_template`
