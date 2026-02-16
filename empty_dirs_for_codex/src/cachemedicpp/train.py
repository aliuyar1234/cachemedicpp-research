"""CLI for CacheMedic++ training with pause/resume and heartbeat monitoring."""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import load_and_validate_config, resolve_repo_root
from .corruption import CorruptionEngine, CorruptionEvent, make_generator
from .data import load_wikitext2
from .hf_patch import enforce_v1_constraints, patch_gpt2_attention
from .io_utils import (
    append_jsonl,
    build_run_identity,
    ensure_run_layout,
    load_run_record,
    utc_now_iso,
    write_json,
    write_run_metadata,
)
from .losses import LayerTriplet, aggregate_objective
from .repair_ops import build_repair_operator
from .runtime import ModelGeometry, load_model_and_tokenizer

TRAIN_STATE_STEP_RE = re.compile(r"train_state_step_(\d+)\.pt$")


def _select_device(device_name: str) -> torch.device:
    name = str(device_name).strip().lower()
    if name.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CacheMedic++ fail-closed: config requested CUDA, "
                "but torch.cuda.is_available() is False."
            )
        return torch.device(name)
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{device_name}'. Use 'cpu' or 'cuda[:index]'.")


def _prepare_token_stream(tokenizer: Any, texts: list[str], *, min_len: int) -> list[int]:
    ids: list[int] = []
    eos = tokenizer.eos_token_id
    for text in texts:
        toks = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not toks:
            continue
        ids.extend(int(t) for t in toks)
        if eos is not None:
            ids.append(int(eos))
    if not ids:
        fallback = [int(eos)] if eos is not None else [0, 1, 2, 3]
        ids = fallback * max(2, min_len)
    while len(ids) < min_len:
        ids.extend(ids[: max(1, min_len - len(ids))])
    return ids


def _sample_prefix(
    token_stream: list[int],
    *,
    prefix_len: int,
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if len(token_stream) <= prefix_len + 1:
        padded = token_stream + token_stream[: prefix_len + 2]
        token_stream = padded
    upper = len(token_stream) - prefix_len - 1
    start = int(torch.randint(0, upper, (1,), generator=gen, device="cpu").item())
    sl = token_stream[start : start + prefix_len]
    return torch.tensor([sl], dtype=torch.long, device=device)


def _build_repair_modules(
    cfg: dict[str, Any],
    geom: ModelGeometry,
    protect_layers: list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.ModuleDict:
    modules = nn.ModuleDict()
    for layer in protect_layers:
        modules[f"layer_{layer}"] = build_repair_operator(
            cfg["repair"],
            head_dim=geom.head_dim,
            num_heads=geom.num_heads,
            layer_index=layer,
            num_layers=geom.num_layers,
        )
    modules.to(device=device, dtype=dtype)
    return modules


def _compute_donor_cache(
    model: Any,
    tokenizer: Any,
    cfg: dict[str, Any],
    protect_layers: list[int],
    *,
    device: torch.device,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    donor_prompt = str(cfg["corruption"]["params"]["donor_prompt"])
    tokens = tokenizer(donor_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if tokens.shape[1] < 2:
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        tokens = torch.tensor([[int(eos), int(eos)]], dtype=torch.long)
    tokens = tokens.to(device)
    with torch.no_grad():
        out = model(input_ids=tokens, use_cache=True)
    pkv = out.past_key_values
    donor: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for layer in protect_layers:
        if layer >= len(pkv):
            continue
        k, v = pkv[layer]
        donor[layer] = (k.detach(), v.detach())
    return donor


def _checkpoint_step(path: Path, pattern: re.Pattern[str]) -> int:
    m = pattern.search(path.name)
    if m is None:
        return -1
    return int(m.group(1))


def _latest_train_state_path(run_dir: Path) -> Path | None:
    latest = run_dir / "checkpoints" / "train_state_latest.pt"
    if latest.exists():
        return latest
    step_states = list((run_dir / "checkpoints").glob("train_state_step_*.pt"))
    if not step_states:
        return None
    step_states.sort(key=lambda p: _checkpoint_step(p, TRAIN_STATE_STEP_RE))
    return step_states[-1]


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device)


def _save_repair_checkpoint(
    *,
    run_dir: Path,
    step: int,
    repair_modules: nn.ModuleDict,
    config_hash: str,
) -> Path:
    ckpt_path = run_dir / "checkpoints" / f"repair_step_{step}.pt"
    torch.save(
        {
            "step": step,
            "repair_state": repair_modules.state_dict(),
            "config_hash": config_hash,
        },
        ckpt_path,
    )
    return ckpt_path


def _save_train_state(
    *,
    run_dir: Path,
    step: int,
    reason: str,
    config_hash: str,
    repair_modules: nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
    data_gen: torch.Generator,
    tokens_since_log: int,
    elapsed_wall_sec: float,
    latest_repair_checkpoint: str | None,
    include_step_file: bool,
    device: torch.device,
) -> tuple[Path, Path | None]:
    payload: dict[str, Any] = {
        "step": int(step),
        "reason": str(reason),
        "timestamp_utc": utc_now_iso(),
        "config_hash": str(config_hash),
        "repair_state": repair_modules.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "data_gen_state": data_gen.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "tokens_since_log": int(tokens_since_log),
        "elapsed_wall_sec": float(elapsed_wall_sec),
        "latest_repair_checkpoint": latest_repair_checkpoint,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

    latest_path = run_dir / "checkpoints" / "train_state_latest.pt"
    torch.save(payload, latest_path)

    step_path: Path | None = None
    if include_step_file:
        step_path = run_dir / "checkpoints" / f"train_state_step_{step}.pt"
        torch.save(payload, step_path)
    return latest_path, step_path


def _restore_train_state(
    *,
    state_path: Path,
    repair_modules: nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
    data_gen: torch.Generator,
    config_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(state_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid train state payload: {state_path}")
    saved_hash = str(payload.get("config_hash", ""))
    if saved_hash and saved_hash != config_hash:
        raise ValueError(
            f"Config hash mismatch for resume. state={saved_hash} current={config_hash}"
        )

    repair_state = payload.get("repair_state")
    if isinstance(repair_state, dict):
        repair_modules.load_state_dict(repair_state, strict=False)

    optimizer_state = payload.get("optimizer_state")
    if isinstance(optimizer_state, dict):
        optimizer.load_state_dict(optimizer_state)
        _optimizer_to_device(optimizer, device)

    data_gen_state = payload.get("data_gen_state")
    if isinstance(data_gen_state, torch.Tensor):
        data_gen.set_state(data_gen_state)

    torch_rng_state = payload.get("torch_rng_state")
    if isinstance(torch_rng_state, torch.Tensor):
        torch.set_rng_state(torch_rng_state)

    if device.type == "cuda" and torch.cuda.is_available():
        cuda_states = payload.get("torch_cuda_rng_state_all")
        if isinstance(cuda_states, list) and cuda_states:
            try:
                torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass

    step = int(payload.get("step", 0))
    return {
        "step": step,
        "tokens_since_log": int(payload.get("tokens_since_log", 0)),
        "elapsed_wall_sec": float(payload.get("elapsed_wall_sec", 0.0)),
        "latest_repair_checkpoint": payload.get("latest_repair_checkpoint"),
        "source_path": str(state_path),
    }


def _make_heartbeat_payload(
    *,
    run_id: str,
    status: str,
    step: int,
    total_steps: int,
    elapsed_wall_sec: float,
    last_tokens_per_sec: float,
    last_gpu_mem_gb: float,
    last_loss_total: float | None,
    latest_repair_checkpoint: str | None,
    latest_train_state: str | None,
    resumed_from: str | None,
    reason: str,
) -> dict[str, Any]:
    step_safe = max(0, int(step))
    total_safe = max(1, int(total_steps))
    progress = min(1.0, float(step_safe / total_safe))
    steps_per_sec = float(step_safe / max(elapsed_wall_sec, 1e-8)) if step_safe > 0 else 0.0
    steps_remaining = max(0, total_safe - step_safe)
    eta_sec = float(steps_remaining / max(steps_per_sec, 1e-8)) if step_safe > 0 else None
    payload: dict[str, Any] = {
        "run_id": run_id,
        "phase": "train",
        "timestamp_utc": utc_now_iso(),
        "status": status,
        "reason": reason,
        "step": step_safe,
        "total_steps": total_safe,
        "progress": progress,
        "steps_remaining": steps_remaining,
        "elapsed_wall_sec": float(elapsed_wall_sec),
        "eta_sec": eta_sec,
        "steps_per_sec": steps_per_sec,
        "tokens_per_sec": float(last_tokens_per_sec),
        "gpu_mem_gb": float(last_gpu_mem_gb),
        "latest_repair_checkpoint": latest_repair_checkpoint,
        "latest_train_state": latest_train_state,
        "resumed_from": resumed_from,
    }
    if last_loss_total is not None:
        payload["loss_total"] = float(last_loss_total)
    return payload


def _emit_heartbeat(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "logs" / "train_heartbeat.json", payload)
    append_jsonl(run_dir / "logs" / "train_heartbeat.jsonl", payload)


def run_train(
    config_path: Path,
    run_dir: Path,
    *,
    resume: bool = True,
    heartbeat_every: int | None = None,
) -> dict[str, Any]:
    repo_root = resolve_repo_root(config_path.parent)
    cfg = load_and_validate_config(config_path, repo_root=repo_root)
    enforce_v1_constraints(cfg, batch_size=int(cfg["train"]["batch_size"]))

    ensure_run_layout(run_dir)
    identity = build_run_identity(run_dir, cfg, config_path)

    run_record = load_run_record(run_dir)
    if run_record is None:
        write_run_metadata(repo_root=repo_root, identity=identity, config_obj=cfg)
    elif str(run_record.get("config_hash", identity.config_hash)) != identity.config_hash:
        raise ValueError(
            "Existing run_record config_hash does not match current config. "
            "Use a different run_dir or disable resume with --no-resume."
        )

    seed = int(cfg["seed"])
    torch.manual_seed(seed)
    device = _select_device(str(cfg["device"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    model, tokenizer, geom, runtime_dtype = load_model_and_tokenizer(cfg, device=device)
    protect_layers = [int(x) for x in cfg["repair"]["protect_layers"]]
    if not protect_layers:
        raise ValueError("repair.protect_layers must not be empty.")
    if max(protect_layers) >= geom.num_layers:
        raise ValueError(
            f"repair.protect_layers contains layer index >= num_layers ({geom.num_layers})."
        )

    repair_modules = _build_repair_modules(
        cfg, geom, protect_layers, device=device, dtype=runtime_dtype
    )
    optimizer = torch.optim.Adam(repair_modules.parameters(), lr=float(cfg["train"]["lr"]))

    engine = CorruptionEngine(config=cfg, protect_layers=protect_layers, repo_root=repo_root)
    donor_cache = _compute_donor_cache(model, tokenizer, cfg, protect_layers, device=device)
    data_gen = make_generator(seed + 17)

    steps = int(cfg["train"]["steps"])
    prefix_len = int(cfg["train"]["prefix_len"])
    log_every = int(cfg["train"]["log_every"])
    save_every = int(cfg["train"]["save_every"])
    if heartbeat_every is None:
        heartbeat_every_steps = max(1, min(log_every, 10))
    else:
        heartbeat_every_steps = int(heartbeat_every)
        if heartbeat_every_steps < 1:
            raise ValueError("--heartbeat_every must be >= 1.")
    clean_batch_prob = float(cfg["train"]["clean_batch_prob"])
    grad_clip = float(cfg["train"]["grad_clip"])
    temperature = float(cfg["train"]["temperature"])
    lambda_id = float(cfg["train"]["lambda_id"])
    lambda_contr = float(cfg["train"]["lambda_contr"])
    alpha_contr = float(cfg["train"]["alpha_contr"])

    texts = load_wikitext2(cfg["data"], num_examples=max(512, steps // 2 + 64), repo_root=repo_root)
    token_stream = _prepare_token_stream(tokenizer, texts, min_len=prefix_len + 2)

    train_log_path = run_dir / "logs" / "train.jsonl"
    control_dir = run_dir / "control"
    pause_request_path = control_dir / "pause.request"
    pause_ack_path = control_dir / "pause_ack.json"

    start_step = 1
    current_step = 0
    elapsed_wall_before = 0.0
    resumed_from: str | None = None
    latest_repair_checkpoint: str | None = None
    tokens_since_log = 0

    if resume:
        state_path = _latest_train_state_path(run_dir)
        if state_path is not None:
            restored = _restore_train_state(
                state_path=state_path,
                repair_modules=repair_modules,
                optimizer=optimizer,
                data_gen=data_gen,
                config_hash=identity.config_hash,
                device=device,
            )
            current_step = int(restored["step"])
            start_step = current_step + 1
            tokens_since_log = int(restored["tokens_since_log"])
            elapsed_wall_before = float(restored["elapsed_wall_sec"])
            resumed_from = str(restored["source_path"])
            latest_repair_checkpoint = (
                str(restored["latest_repair_checkpoint"])
                if restored.get("latest_repair_checkpoint")
                else None
            )

    train_start_wall = time.perf_counter()
    last_log_time = train_start_wall
    last_tokens_per_sec = 0.0
    last_gpu_mem = 0.0
    last_loss_total: float | None = None
    latest_train_state_path: str | None = None

    initial_status = "resumed" if resumed_from is not None else "running"
    initial_reason = "startup_resume" if resumed_from is not None else "startup"
    _emit_heartbeat(
        run_dir,
        _make_heartbeat_payload(
            run_id=identity.run_id,
            status=initial_status,
            step=current_step,
            total_steps=steps,
            elapsed_wall_sec=elapsed_wall_before,
            last_tokens_per_sec=last_tokens_per_sec,
            last_gpu_mem_gb=last_gpu_mem,
            last_loss_total=last_loss_total,
            latest_repair_checkpoint=latest_repair_checkpoint,
            latest_train_state=latest_train_state_path,
            resumed_from=resumed_from,
            reason=initial_reason,
        ),
    )

    if start_step > steps:
        completed_payload = _make_heartbeat_payload(
            run_id=identity.run_id,
            status="completed",
            step=steps,
            total_steps=steps,
            elapsed_wall_sec=elapsed_wall_before,
            last_tokens_per_sec=last_tokens_per_sec,
            last_gpu_mem_gb=last_gpu_mem,
            last_loss_total=last_loss_total,
            latest_repair_checkpoint=latest_repair_checkpoint,
            latest_train_state=latest_train_state_path,
            resumed_from=resumed_from,
            reason="already_completed",
        )
        _emit_heartbeat(run_dir, completed_payload)
        return {
            "run_id": identity.run_id,
            "run_dir": str(run_dir),
            "steps": steps,
            "status": "completed",
            "already_completed": True,
            "resumed": resumed_from is not None,
        }

    def pause_and_return(reason: str) -> dict[str, Any]:
        nonlocal latest_repair_checkpoint
        nonlocal latest_train_state_path

        if pause_request_path.exists():
            pause_request_path.unlink(missing_ok=True)
        elapsed_wall = elapsed_wall_before + (time.perf_counter() - train_start_wall)

        if current_step > 0:
            ckpt_path = _save_repair_checkpoint(
                run_dir=run_dir,
                step=current_step,
                repair_modules=repair_modules,
                config_hash=identity.config_hash,
            )
            latest_repair_checkpoint = str(ckpt_path)

        latest_state, step_state = _save_train_state(
            run_dir=run_dir,
            step=current_step,
            reason=reason,
            config_hash=identity.config_hash,
            repair_modules=repair_modules,
            optimizer=optimizer,
            data_gen=data_gen,
            tokens_since_log=tokens_since_log,
            elapsed_wall_sec=elapsed_wall,
            latest_repair_checkpoint=latest_repair_checkpoint,
            include_step_file=True,
            device=device,
        )
        latest_train_state_path = str(latest_state)

        write_json(
            pause_ack_path,
            {
                "run_id": identity.run_id,
                "status": "paused",
                "reason": reason,
                "step": current_step,
                "timestamp_utc": utc_now_iso(),
                "resume_state": str(step_state or latest_state),
            },
        )
        append_jsonl(
            train_log_path,
            {
                "run_id": identity.run_id,
                "phase": "train_control",
                "timestamp_utc": utc_now_iso(),
                "event": "paused",
                "reason": reason,
                "step": current_step,
                "resume_state": str(step_state or latest_state),
            },
        )
        _emit_heartbeat(
            run_dir,
            _make_heartbeat_payload(
                run_id=identity.run_id,
                status="paused",
                step=current_step,
                total_steps=steps,
                elapsed_wall_sec=elapsed_wall,
                last_tokens_per_sec=last_tokens_per_sec,
                last_gpu_mem_gb=last_gpu_mem,
                last_loss_total=last_loss_total,
                latest_repair_checkpoint=latest_repair_checkpoint,
                latest_train_state=str(step_state or latest_state),
                resumed_from=resumed_from,
                reason=reason,
            ),
        )
        return {
            "run_id": identity.run_id,
            "run_dir": str(run_dir),
            "steps": steps,
            "last_step": current_step,
            "status": "paused",
            "pause_reason": reason,
            "resume_state": str(step_state or latest_state),
            "resumed": resumed_from is not None,
        }

    protect_set = set(protect_layers)
    try:
        for step in range(start_step, steps + 1):
            if pause_request_path.exists():
                return pause_and_return("pause_request")

            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)

            input_ids = _sample_prefix(
                token_stream, prefix_len=prefix_len, gen=data_gen, device=device
            )
            if int(input_ids.shape[0]) != 1:
                raise ValueError("CacheMedic++ v1 fail-closed: bsz must be 1 in training loop.")

            clean_draw = torch.rand((1,), generator=data_gen, device="cpu").item()
            clean_mode = clean_draw < clean_batch_prob
            if clean_mode:
                event = CorruptionEvent(corruption_type="none", epsilon=0.0)
                corruption_active = False
            else:
                event = engine.sample_event()
                corruption_active = event.corruption_type != "none"

            with torch.no_grad():
                teacher_logits = model(input_ids=input_ids, use_cache=True).logits[:, -1, :].float()

            triplets: list[LayerTriplet] = []
            head_counts: list[int] = []
            time_counts: list[int] = []

            def student_patch(
                layer_idx: int, q: torch.Tensor, key: torch.Tensor, value: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                if layer_idx not in protect_set:
                    return key, value
                K_clean = key.detach()
                V_clean = value.detach()
                if corruption_active:
                    K_corr, V_corr, summary = engine.apply_to_layer(
                        layer_index=layer_idx,
                        K=K_clean,
                        V=V_clean,
                        event=event,
                        donor=donor_cache.get(layer_idx),
                        return_masks=False,
                    )
                else:
                    K_corr, V_corr = K_clean, V_clean
                    summary = {"head_count": 0, "time_count": 0}

                module = repair_modules[f"layer_{layer_idx}"]
                K_rep, V_rep = module(q.detach(), K_corr, V_corr)
                triplets.append(
                    LayerTriplet(
                        layer=layer_idx,
                        K_clean=K_clean,
                        V_clean=V_clean,
                        K_corr=K_corr,
                        V_corr=V_corr,
                        K_rep=K_rep,
                        V_rep=V_rep,
                    )
                )
                head_counts.append(int(summary.get("head_count", 0)))
                time_counts.append(int(summary.get("time_count", 0)))
                return K_rep, V_rep

            handle = patch_gpt2_attention(model, student_patch, protect_layers=protect_layers)
            try:
                student_logits = model(input_ids=input_ids, use_cache=True).logits[:, -1, :].float()
            finally:
                handle.unpatch()

            objective = aggregate_objective(
                z_clean=teacher_logits,
                z_rep=student_logits,
                layer_triplets=triplets,
                temperature=temperature,
                lambda_id=lambda_id,
                lambda_contr=lambda_contr,
                alpha_contr=alpha_contr,
                corruption_active=corruption_active,
                apply_to=str(cfg["repair"].get("apply_to", "KV")),
                repair_parameters=[p for p in repair_modules.parameters() if p.requires_grad],
            )

            loss = objective["loss_total"]
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(repair_modules.parameters(), grad_clip)
            optimizer.step()

            current_step = step
            tokens_since_log += int(input_ids.shape[1])
            last_loss_total = float(objective["loss_total"].detach().item())

            needs_log = step % log_every == 0 or step == steps
            needs_ckpt = step % save_every == 0 or step == steps
            needs_heartbeat = (
                step % heartbeat_every_steps == 0 or needs_log or needs_ckpt or step == steps
            )

            if needs_log:
                now = time.perf_counter()
                dt = max(now - last_log_time, 1e-8)
                last_tokens_per_sec = float(tokens_since_log / dt)
                tokens_since_log = 0
                last_log_time = now
                last_gpu_mem = 0.0
                if device.type == "cuda":
                    last_gpu_mem = float(torch.cuda.max_memory_allocated(device) / (1024**3))
                    torch.cuda.reset_peak_memory_stats(device)

                append_jsonl(
                    train_log_path,
                    {
                        "run_id": identity.run_id,
                        "phase": "train",
                        "timestamp_utc": utc_now_iso(),
                        "step": step,
                        "loss_total": float(objective["loss_total"].detach().item()),
                        "loss_kd": float(objective["loss_kd"].detach().item()),
                        "loss_id": float(objective["loss_id"].detach().item()),
                        "loss_contr": float(objective["loss_contr"].detach().item()),
                        "rho_mean": float(objective["rho_mean"]),
                        "rho_p95": float(objective["rho_p95"]),
                        "corruption": {
                            **event.as_log_dict(),
                            "head_count_mean": float(sum(head_counts) / max(1, len(head_counts))),
                            "time_count_mean": float(sum(time_counts) / max(1, len(time_counts))),
                        },
                        "tokens_per_sec": last_tokens_per_sec,
                        "gpu_mem_gb": last_gpu_mem,
                        "step_wall_sec": float(time.perf_counter() - t0),
                    },
                )

            if needs_ckpt:
                ckpt_path = _save_repair_checkpoint(
                    run_dir=run_dir,
                    step=step,
                    repair_modules=repair_modules,
                    config_hash=identity.config_hash,
                )
                latest_repair_checkpoint = str(ckpt_path)

            if needs_heartbeat:
                elapsed_wall = elapsed_wall_before + (time.perf_counter() - train_start_wall)
                latest_state, step_state = _save_train_state(
                    run_dir=run_dir,
                    step=step,
                    reason="periodic_checkpoint" if needs_ckpt else "heartbeat",
                    config_hash=identity.config_hash,
                    repair_modules=repair_modules,
                    optimizer=optimizer,
                    data_gen=data_gen,
                    tokens_since_log=tokens_since_log,
                    elapsed_wall_sec=elapsed_wall,
                    latest_repair_checkpoint=latest_repair_checkpoint,
                    include_step_file=needs_ckpt,
                    device=device,
                )
                latest_train_state_path = str(step_state or latest_state)
                _emit_heartbeat(
                    run_dir,
                    _make_heartbeat_payload(
                        run_id=identity.run_id,
                        status="running",
                        step=step,
                        total_steps=steps,
                        elapsed_wall_sec=elapsed_wall,
                        last_tokens_per_sec=last_tokens_per_sec,
                        last_gpu_mem_gb=last_gpu_mem,
                        last_loss_total=last_loss_total,
                        latest_repair_checkpoint=latest_repair_checkpoint,
                        latest_train_state=latest_train_state_path,
                        resumed_from=resumed_from,
                        reason="heartbeat",
                    ),
                )

            if pause_request_path.exists():
                return pause_and_return("pause_request")

    except KeyboardInterrupt:
        return pause_and_return("keyboard_interrupt")

    elapsed_wall = elapsed_wall_before + (time.perf_counter() - train_start_wall)
    _emit_heartbeat(
        run_dir,
        _make_heartbeat_payload(
            run_id=identity.run_id,
            status="completed",
            step=steps,
            total_steps=steps,
            elapsed_wall_sec=elapsed_wall,
            last_tokens_per_sec=last_tokens_per_sec,
            last_gpu_mem_gb=last_gpu_mem,
            last_loss_total=last_loss_total,
            latest_repair_checkpoint=latest_repair_checkpoint,
            latest_train_state=latest_train_state_path,
            resumed_from=resumed_from,
            reason="completed",
        ),
    )
    append_jsonl(
        train_log_path,
        {
            "run_id": identity.run_id,
            "phase": "train_control",
            "timestamp_utc": utc_now_iso(),
            "event": "completed",
            "step": steps,
        },
    )
    return {
        "run_id": identity.run_id,
        "run_dir": str(run_dir),
        "steps": steps,
        "last_step": steps,
        "status": "completed",
        "resumed": resumed_from is not None,
        "resumed_from": resumed_from,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train CacheMedic++ repair operator.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--run_dir", required=True, help="Output run directory.")
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from latest train state checkpoint if available (default).",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start from step 1 even if resume state checkpoint exists.",
    )
    p.set_defaults(resume=True)
    p.add_argument(
        "--heartbeat_every",
        type=int,
        default=None,
        help="Heartbeat interval in steps (default: min(train.log_every, 10)).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_train(
        Path(args.config).resolve(),
        Path(args.run_dir).resolve(),
        resume=bool(args.resume),
        heartbeat_every=args.heartbeat_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
