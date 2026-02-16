"""CLI for CacheMedic++ task robustness evaluation with real logits."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from .config import load_and_validate_config, resolve_repo_root, stable_json_hash
from .corruption import CorruptionEngine, CorruptionEvent, auc, make_generator, mask_ht, rms
from .data import (
    SST2_LABEL_NEG,
    SST2_LABEL_POS,
    build_sst2_prompt,
    generate_needle_dataset,
    load_sst2,
    load_wikitext2,
    normalize_answer,
)
from .hf_patch import enforce_v1_constraints, patch_gpt2_attention
from .io_utils import append_jsonl, ensure_run_layout, load_run_record, utc_now_iso, write_json
from .repair_ops import build_repair_operator
from .runtime import ModelGeometry, load_model_and_tokenizer

HEURISTICS = ["reset_clear", "smoothing", "masking_dropout", "clipping_renorm"]
EPS0 = 1e-8
EVAL_PROGRESS_PATH = Path("metrics") / "eval_progress.json"


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


def _to_score(task: str, value: float) -> float:
    if task == "wikitext2_ppl":
        return 1.0 / max(value, 1e-8)
    return value


def _mean(xs: list[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def _unit_key(
    *,
    stage: str,
    task: str,
    defense: str,
    corruption_type: str,
    epsilon: float,
) -> str:
    return "|".join(
        [
            stage,
            task,
            defense,
            corruption_type,
            f"{float(epsilon):.12g}",
        ]
    )


def _pause_request_path(run_dir: Path) -> Path:
    return run_dir / "control" / "pause.request"


def _pause_requested(run_dir: Path) -> bool:
    return _pause_request_path(run_dir).exists()


def _eval_progress_file(run_dir: Path) -> Path:
    return run_dir / EVAL_PROGRESS_PATH


def _load_eval_progress(run_dir: Path) -> dict[str, Any] | None:
    path = _eval_progress_file(run_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _save_eval_progress(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(_eval_progress_file(run_dir), payload)


def _make_eval_heartbeat(
    *,
    run_id: str,
    status: str,
    step: int,
    total_steps: int,
    elapsed_wall_sec: float,
    reason: str,
    resumed_from: str | None,
    current_unit: dict[str, Any] | None,
) -> dict[str, Any]:
    done = max(0, int(step))
    total = max(1, int(total_steps))
    progress = min(1.0, float(done / total))
    units_per_sec = float(done / max(elapsed_wall_sec, 1e-8)) if done > 0 else 0.0
    remaining = max(0, total - done)
    eta = float(remaining / max(units_per_sec, 1e-8)) if done > 0 else None
    payload: dict[str, Any] = {
        "phase": "eval",
        "run_id": run_id,
        "timestamp_utc": utc_now_iso(),
        "status": status,
        "reason": reason,
        "step": done,
        "total_steps": total,
        "progress": progress,
        "steps_remaining": remaining,
        "elapsed_wall_sec": float(elapsed_wall_sec),
        "eta_sec": eta,
        "steps_per_sec": units_per_sec,
        "resumed_from": resumed_from,
    }
    if current_unit is not None:
        payload["current_unit"] = current_unit
    return payload


def _emit_eval_heartbeat(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "logs" / "eval_heartbeat.json", payload)
    append_jsonl(run_dir / "logs" / "eval_heartbeat.jsonl", payload)


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
    modules.eval()
    return modules


def _latest_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    pats = list(ckpt_dir.glob("repair_step_*.pt"))
    if not pats:
        return None

    def step_of(path: Path) -> int:
        m = re.search(r"repair_step_(\d+)\.pt$", path.name)
        return int(m.group(1)) if m else -1

    pats.sort(key=step_of)
    return pats[-1]


def _load_repair_state(run_dir: Path, modules: nn.ModuleDict) -> bool:
    ckpt = _latest_checkpoint(run_dir)
    if ckpt is None:
        return False
    payload = torch.load(ckpt, map_location="cpu")
    state = payload.get("repair_state")
    if not isinstance(state, dict):
        return False
    modules.load_state_dict(state, strict=False)
    return True


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
        if layer < len(pkv):
            k, v = pkv[layer]
            donor[layer] = (k.detach(), v.detach())
    return donor


def _apply_heuristic(
    defense: str,
    *,
    K: torch.Tensor,
    V: torch.Tensor,
    head_mask: torch.Tensor,
    time_mask: torch.Tensor,
    baseline_cfg: dict[str, Any],
    gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if defense == "reset_clear":
        M = mask_ht(head_mask, time_mask, int(K.shape[1]), int(K.shape[2]), K.device).expand_as(K)
        zK = torch.zeros_like(K)
        zV = torch.zeros_like(V)
        return torch.where(M, zK, K), torch.where(M, zV, V)

    if defense == "smoothing":
        beta = float(baseline_cfg["smoothing_beta"])
        K2, V2 = K.clone(), V.clone()
        H = int(K.shape[1])
        T = int(K.shape[2])
        hm = head_mask.view(1, H, 1).to(device=K.device, dtype=torch.bool)
        for t in range(1, T):
            if bool(time_mask[t].item()):
                k_sm = (1.0 - beta) * K2[:, :, t, :] + beta * K2[:, :, t - 1, :]
                v_sm = (1.0 - beta) * V2[:, :, t, :] + beta * V2[:, :, t - 1, :]
                K2[:, :, t, :] = torch.where(hm, k_sm, K2[:, :, t, :])
                V2[:, :, t, :] = torch.where(hm, v_sm, V2[:, :, t, :])
        return K2, V2

    if defense == "masking_dropout":
        p = float(baseline_cfg["dropout_p"])
        M = mask_ht(head_mask, time_mask, int(K.shape[1]), int(K.shape[2]), K.device).expand_as(K)
        drop = torch.rand(K.shape, generator=gen, device="cpu", dtype=torch.float32).to(K.device) < p
        Mfull = M & drop
        zK = torch.zeros_like(K)
        zV = torch.zeros_like(V)
        return torch.where(Mfull, zK, K), torch.where(Mfull, zV, V)

    if defense == "clipping_renorm":
        clip_val = float(baseline_cfg["clip_val"])
        target_r = float(baseline_cfg["target_rms"])
        M = mask_ht(head_mask, time_mask, int(K.shape[1]), int(K.shape[2]), K.device)
        Kc = torch.clamp(K, -clip_val, clip_val)
        Vc = torch.clamp(V, -clip_val, clip_val)
        Kr = Kc * (target_r / (rms(Kc) + EPS0))
        Vr = Vc * (target_r / (rms(Vc) + EPS0))
        return torch.where(M, Kr, K), torch.where(M, Vr, V)

    raise ValueError(f"Unknown heuristic defense: {defense}")


def _forward_with_patch(
    model: Any,
    input_ids: torch.Tensor,
    *,
    patch_fn: Callable[
        [int, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ]
    | None,
    protect_layers: list[int],
) -> torch.Tensor:
    if patch_fn is None:
        with torch.no_grad():
            return model(input_ids=input_ids, use_cache=True).logits
    handle = patch_gpt2_attention(model, patch_fn, protect_layers=protect_layers)
    try:
        with torch.no_grad():
            return model(input_ids=input_ids, use_cache=True).logits
    finally:
        handle.unpatch()


def _sequence_logprob(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
    device: torch.device,
    patch_fn: Callable[
        [int, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ]
    | None,
    protect_layers: list[int],
    max_seq_len: int,
) -> tuple[float, int]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not comp_ids:
        return 0.0, 0
    if not prompt_ids:
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        prompt_ids = [int(eos)]

    all_ids = prompt_ids + comp_ids
    if len(all_ids) < 2:
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        all_ids = [int(eos), int(eos)]

    if len(all_ids) > max_seq_len + 1:
        all_ids = all_ids[-(max_seq_len + 1) :]
        prompt_keep = max(1, len(all_ids) - len(comp_ids))
    else:
        prompt_keep = len(prompt_ids)

    input_ids = torch.tensor([all_ids[:-1]], dtype=torch.long, device=device)
    target_ids = torch.tensor(all_ids[1:], dtype=torch.long, device=device)
    logits = _forward_with_patch(model, input_ids, patch_fn=patch_fn, protect_layers=protect_layers)[0]
    log_probs = F.log_softmax(logits, dim=-1)

    start = max(0, prompt_keep - 1)
    count = min(len(comp_ids), int(target_ids.shape[0]) - start)
    if count <= 0:
        return 0.0, 0
    idx = torch.arange(start, start + count, device=device)
    tok = target_ids[idx]
    lp = log_probs[idx, tok].sum()
    return float(lp.item()), count


def _evaluate_wikitext_ppl(
    model: Any,
    tokenizer: Any,
    *,
    texts: list[str],
    max_seq_len: int,
    device: torch.device,
    patch_fn: Callable[
        [int, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ]
    | None,
    protect_layers: list[int],
) -> tuple[float, int]:
    total_nll = 0.0
    total_tok = 0
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_seq_len + 1)[
            "input_ids"
        ]
        if len(ids) < 2:
            continue
        input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        target = torch.tensor(ids[1:], dtype=torch.long, device=device)
        logits = _forward_with_patch(model, input_ids, patch_fn=patch_fn, protect_layers=protect_layers)[0]
        nll = F.cross_entropy(logits, target, reduction="sum")
        total_nll += float(nll.item())
        total_tok += int(target.numel())
    if total_tok == 0:
        return float("inf"), 0
    ppl = math.exp(min(80.0, total_nll / total_tok))
    return float(ppl), total_tok


def _evaluate_sst2_accuracy(
    model: Any,
    tokenizer: Any,
    *,
    rows: list[dict[str, Any]],
    max_seq_len: int,
    device: torch.device,
    patch_fn: Callable[
        [int, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ]
    | None,
    protect_layers: list[int],
) -> tuple[float, int]:
    correct = 0
    total = 0
    token_counter = 0
    for row in rows:
        prompt = build_sst2_prompt(str(row["text"]))
        lp_neg, tneg = _sequence_logprob(
            model,
            tokenizer,
            prompt=prompt,
            completion=SST2_LABEL_NEG,
            device=device,
            patch_fn=patch_fn,
            protect_layers=protect_layers,
            max_seq_len=max_seq_len,
        )
        lp_pos, tpos = _sequence_logprob(
            model,
            tokenizer,
            prompt=prompt,
            completion=SST2_LABEL_POS,
            device=device,
            patch_fn=patch_fn,
            protect_layers=protect_layers,
            max_seq_len=max_seq_len,
        )
        pred = 1 if lp_pos >= lp_neg else 0
        correct += int(pred == int(row["label"]))
        total += 1
        token_counter += tneg + tpos
    if total == 0:
        return 0.0, 0
    return float(correct / total), token_counter


def _evaluate_needle_accuracy(
    model: Any,
    tokenizer: Any,
    *,
    rows: list[dict[str, str]],
    max_seq_len: int,
    device: torch.device,
    patch_fn: Callable[
        [int, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ]
    | None,
    protect_layers: list[int],
) -> tuple[float, int]:
    correct = 0
    total = 0
    token_counter = 0
    for row in rows:
        prompt = f"{row['context']}\n{row['question']}"
        answer = str(row["answer"])
        candidates = [f" {answer}", " unknown", " none", " violet-6"]
        best_lp = -float("inf")
        best_cand = ""
        for cand in candidates:
            lp, tok = _sequence_logprob(
                model,
                tokenizer,
                prompt=prompt,
                completion=cand,
                device=device,
                patch_fn=patch_fn,
                protect_layers=protect_layers,
                max_seq_len=max_seq_len,
            )
            token_counter += tok
            if lp > best_lp:
                best_lp = lp
                best_cand = cand
        if normalize_answer(best_cand) == normalize_answer(answer):
            correct += 1
        total += 1
    if total == 0:
        return 0.0, 0
    return float(correct / total), token_counter


def run_eval(
    config_path: Path,
    run_dir: Path,
    *,
    resume: bool = True,
    heartbeat_every: int = 1,
) -> dict[str, Any]:
    repo_root = resolve_repo_root(config_path.parent)
    cfg = load_and_validate_config(config_path, repo_root=repo_root)
    enforce_v1_constraints(cfg, batch_size=1)
    ensure_run_layout(run_dir)
    if int(heartbeat_every) < 1:
        raise ValueError("heartbeat_every must be >= 1.")

    run_record = load_run_record(run_dir)
    run_id = str(run_record.get("run_id")) if run_record else run_dir.name
    config_hash = stable_json_hash(cfg)
    progress_path = _eval_progress_file(run_dir)
    pause_request_path = _pause_request_path(run_dir)

    device = _select_device(str(cfg["device"]))
    if device.type == "cuda":
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
    loaded_ckpt = _load_repair_state(run_dir, repair_modules)
    engine = CorruptionEngine(config=cfg, protect_layers=protect_layers, repo_root=repo_root)
    donor_cache = _compute_donor_cache(model, tokenizer, cfg, protect_layers, device=device)
    rng = make_generator(int(cfg["seed"]) + 333)
    protect_set = set(protect_layers)

    task_specs = cfg["eval"]["tasks"]
    eps_grid = [float(x) for x in cfg["eval"]["corruption_eval"]["eps"]]
    corr_types = [str(x) for x in cfg["eval"]["corruption_eval"]["types"]]
    defenses = ["no_defense", *HEURISTICS, "cachemedic"]
    max_seq_len = int(cfg["eval"]["max_seq_len"])
    total_units = len(task_specs) * len(defenses) * (1 + (len(corr_types) * len(eps_grid)))

    if resume and (run_dir / "metrics" / "task_metrics.json").exists() and not progress_path.exists():
        _emit_eval_heartbeat(
            run_dir,
            _make_eval_heartbeat(
                run_id=run_id,
                status="completed",
                step=total_units,
                total_steps=total_units,
                elapsed_wall_sec=0.0,
                reason="already_completed",
                resumed_from=None,
                current_unit=None,
            ),
        )
        return {
            "run_id": run_id,
            "status": "completed",
            "already_completed": True,
        }

    task_data: dict[str, Any] = {}
    for task in task_specs:
        name = str(task["name"])
        n = int(task["num_examples"])
        if name == "wikitext2_ppl":
            task_data[name] = load_wikitext2(cfg["data"], num_examples=n, repo_root=repo_root)
        elif name == "sst2_prompted":
            task_data[name] = load_sst2(cfg["data"], num_examples=n, repo_root=repo_root)
        elif name == "needle_long_context":
            task_data[name] = generate_needle_dataset(cfg["data"]["long_context"], n)
        else:
            raise ValueError(f"Unsupported eval task: {name}")

    overhead_raw: dict[str, dict[str, float]] = {
        d: {"tokens": 0.0, "time": 0.0, "gpu_mem_gb": 0.0} for d in defenses
    }

    def make_patch(defense: str, event: CorruptionEvent, corruption_active: bool):
        if defense == "no_defense" and not corruption_active:
            return None

        def patch(
            layer_idx: int, q: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if layer_idx not in protect_set:
                return key, value

            if corruption_active:
                K_corr, V_corr, summary = engine.apply_to_layer(
                    layer_index=layer_idx,
                    K=key,
                    V=value,
                    event=event,
                    donor=donor_cache.get(layer_idx),
                    return_masks=True,
                )
                head_mask = summary["head_mask"]
                time_mask = summary["time_mask"]
            else:
                K_corr, V_corr = key, value
                head_mask = torch.zeros((int(key.shape[1]),), dtype=torch.bool)
                time_mask = torch.zeros((int(key.shape[2]),), dtype=torch.bool)

            if defense == "cachemedic":
                module = repair_modules[f"layer_{layer_idx}"]
                return module(q.detach(), K_corr.detach(), V_corr.detach())

            if defense in HEURISTICS and corruption_active:
                return _apply_heuristic(
                    defense,
                    K=K_corr,
                    V=V_corr,
                    head_mask=head_mask,
                    time_mask=time_mask,
                    baseline_cfg=cfg["baseline"],
                    gen=rng,
                )
            return K_corr, V_corr

        return patch

    def evaluate_task(
        task_name: str,
        defense: str,
        event: CorruptionEvent,
        corruption_active: bool,
    ) -> tuple[float, int, float]:
        patch_fn = make_patch(defense, event, corruption_active)
        start = time.perf_counter()
        if task_name == "wikitext2_ppl":
            value, tokens = _evaluate_wikitext_ppl(
                model,
                tokenizer,
                texts=task_data[task_name],
                max_seq_len=max_seq_len,
                device=device,
                patch_fn=patch_fn,
                protect_layers=protect_layers,
            )
        elif task_name == "sst2_prompted":
            value, tokens = _evaluate_sst2_accuracy(
                model,
                tokenizer,
                rows=task_data[task_name],
                max_seq_len=max_seq_len,
                device=device,
                patch_fn=patch_fn,
                protect_layers=protect_layers,
            )
        elif task_name == "needle_long_context":
            value, tokens = _evaluate_needle_accuracy(
                model,
                tokenizer,
                rows=task_data[task_name],
                max_seq_len=max_seq_len,
                device=device,
                patch_fn=patch_fn,
                protect_layers=protect_layers,
            )
        else:
            raise ValueError(task_name)
        elapsed = time.perf_counter() - start
        return value, tokens, elapsed

    work_items: list[dict[str, Any]] = []
    clean_event = CorruptionEvent(corruption_type="none", epsilon=0.0)
    for task in task_specs:
        task_name = str(task["name"])
        for defense in defenses:
            work_items.append(
                {
                    "stage": "clean",
                    "task": task_name,
                    "corruption_type": "none",
                    "epsilon": 0.0,
                    "defense": defense,
                    "corruption_active": False,
                    "unit_key": _unit_key(
                        stage="clean",
                        task=task_name,
                        defense=defense,
                        corruption_type="none",
                        epsilon=0.0,
                    ),
                }
            )
        for ctype in corr_types:
            for eps in eps_grid:
                for defense in defenses:
                    work_items.append(
                        {
                            "stage": "corr",
                            "task": task_name,
                            "corruption_type": str(ctype),
                            "epsilon": float(eps),
                            "defense": defense,
                            "corruption_active": float(eps) > 0.0,
                            "unit_key": _unit_key(
                                stage="corr",
                                task=task_name,
                                defense=defense,
                                corruption_type=str(ctype),
                                epsilon=float(eps),
                            ),
                        }
                    )

    progress = _load_eval_progress(run_dir) if resume else None
    records: list[dict[str, Any]] = []
    completed_unit_keys: set[str] = set()
    resumed_from: str | None = None
    elapsed_before = 0.0
    created_utc = utc_now_iso()

    if progress is not None:
        saved_hash = str(progress.get("config_hash", ""))
        if saved_hash and saved_hash != config_hash:
            raise ValueError(
                f"Config hash mismatch for eval resume. state={saved_hash} current={config_hash}"
            )
        recs = progress.get("records", [])
        if isinstance(recs, list):
            for rec in recs:
                if isinstance(rec, dict) and isinstance(rec.get("unit_key"), str):
                    records.append(rec)
                    completed_unit_keys.add(str(rec["unit_key"]))
        elapsed_before = float(progress.get("elapsed_wall_sec", 0.0))
        created_utc = str(progress.get("created_utc", created_utc))
        resumed_from = str(progress_path)

        if (
            str(progress.get("status", "")) == "completed"
            and (run_dir / "metrics" / "task_metrics.json").exists()
        ):
            _emit_eval_heartbeat(
                run_dir,
                _make_eval_heartbeat(
                    run_id=run_id,
                    status="completed",
                    step=total_units,
                    total_steps=total_units,
                    elapsed_wall_sec=elapsed_before,
                    reason="already_completed",
                    resumed_from=resumed_from,
                    current_unit=None,
                ),
            )
            return {
                "run_id": run_id,
                "status": "completed",
                "already_completed": True,
                "resumed": True,
            }

    start_wall = time.perf_counter()

    def persist_progress(
        *,
        status: str,
        reason: str,
        current_unit: dict[str, Any] | None,
    ) -> float:
        elapsed_total = elapsed_before + (time.perf_counter() - start_wall)
        payload = {
            "phase": "eval",
            "run_id": run_id,
            "config_hash": config_hash,
            "status": status,
            "reason": reason,
            "created_utc": created_utc,
            "updated_utc": utc_now_iso(),
            "total_units": total_units,
            "completed_units": len(completed_unit_keys),
            "elapsed_wall_sec": float(elapsed_total),
            "resumed_from": resumed_from,
            "records": records,
        }
        _save_eval_progress(run_dir, payload)
        _emit_eval_heartbeat(
            run_dir,
            _make_eval_heartbeat(
                run_id=run_id,
                status=status,
                step=len(completed_unit_keys),
                total_steps=total_units,
                elapsed_wall_sec=elapsed_total,
                reason=reason,
                resumed_from=resumed_from,
                current_unit=current_unit,
            ),
        )
        return elapsed_total

    def pause_and_return(reason: str, current_unit: dict[str, Any] | None) -> dict[str, Any]:
        pause_request_path.unlink(missing_ok=True)
        elapsed_total = persist_progress(status="paused", reason=reason, current_unit=current_unit)
        write_json(
            run_dir / "control" / "pause_ack_eval.json",
            {
                "phase": "eval",
                "run_id": run_id,
                "status": "paused",
                "reason": reason,
                "timestamp_utc": utc_now_iso(),
                "resume_state": str(progress_path),
                "completed_units": len(completed_unit_keys),
                "total_units": total_units,
                "elapsed_wall_sec": float(elapsed_total),
            },
        )
        return {
            "run_id": run_id,
            "status": "paused",
            "pause_reason": reason,
            "resume_state": str(progress_path),
            "completed_units": len(completed_unit_keys),
            "total_units": total_units,
            "resumed": resumed_from is not None,
        }

    startup_reason = "startup_resume" if resumed_from is not None else "startup"
    startup_status = "resumed" if resumed_from is not None else "running"
    _emit_eval_heartbeat(
        run_dir,
        _make_eval_heartbeat(
            run_id=run_id,
            status=startup_status,
            step=len(completed_unit_keys),
            total_steps=total_units,
            elapsed_wall_sec=elapsed_before,
            reason=startup_reason,
            resumed_from=resumed_from,
            current_unit=None,
        ),
    )

    if len(completed_unit_keys) >= total_units and (run_dir / "metrics" / "task_metrics.json").exists():
        persist_progress(status="completed", reason="already_completed", current_unit=None)
        return {
            "run_id": run_id,
            "status": "completed",
            "already_completed": True,
            "resumed": resumed_from is not None,
        }

    units_since_heartbeat = 0
    for unit in work_items:
        unit_key = str(unit["unit_key"])
        if unit_key in completed_unit_keys:
            continue

        if _pause_requested(run_dir):
            return pause_and_return("pause_request", unit)

        task_name = str(unit["task"])
        defense = str(unit["defense"])
        eps = float(unit["epsilon"])
        ctype = str(unit["corruption_type"])
        stage = str(unit["stage"])
        event = clean_event if stage == "clean" else CorruptionEvent(corruption_type=ctype, epsilon=eps)
        value, tokens, elapsed = evaluate_task(
            task_name,
            defense,
            event,
            bool(unit["corruption_active"]),
        )
        gpu_mem_gb = 0.0
        if device.type == "cuda":
            gpu_mem_gb = float(torch.cuda.max_memory_allocated(device) / (1024**3))
            torch.cuda.reset_peak_memory_stats(device)

        rec = {
            "unit_key": unit_key,
            "stage": stage,
            "task": task_name,
            "corruption_type": ctype,
            "epsilon": eps,
            "defense": defense,
            "value": float(value),
            "score": float(_to_score(task_name, value)),
            "tokens": int(tokens),
            "elapsed_sec": float(elapsed),
            "gpu_mem_gb": float(gpu_mem_gb),
        }
        records.append(rec)
        completed_unit_keys.add(unit_key)
        units_since_heartbeat += 1

        if units_since_heartbeat >= int(heartbeat_every) or len(completed_unit_keys) == total_units:
            persist_progress(status="running", reason="progress", current_unit=unit)
            units_since_heartbeat = 0

    persist_progress(status="running", reason="finalizing", current_unit=None)

    record_by_key: dict[str, dict[str, Any]] = {}
    for rec in records:
        key = str(rec.get("unit_key", ""))
        if key:
            record_by_key[key] = rec

    ordered_records: list[dict[str, Any]] = []
    for unit in work_items:
        key = str(unit["unit_key"])
        if key not in record_by_key:
            raise RuntimeError(f"Missing eval unit record: {key}")
        ordered_records.append(record_by_key[key])

    clean_metrics: dict[str, dict[str, float]] = {
        str(task["name"]): {} for task in task_specs
    }
    curves: list[dict[str, Any]] = []
    overhead_raw = {d: {"tokens": 0.0, "time": 0.0, "gpu_mem_gb": 0.0} for d in defenses}
    for rec in ordered_records:
        task_name = str(rec["task"])
        defense = str(rec["defense"])
        if defense not in overhead_raw:
            continue
        overhead_raw[defense]["tokens"] += float(rec["tokens"])
        overhead_raw[defense]["time"] += float(rec["elapsed_sec"])
        overhead_raw[defense]["gpu_mem_gb"] = max(
            overhead_raw[defense]["gpu_mem_gb"], float(rec["gpu_mem_gb"])
        )
        if str(rec["stage"]) == "clean":
            clean_metrics[task_name][defense] = float(rec["value"])
        else:
            curves.append(
                {
                    "task": task_name,
                    "corruption_type": str(rec["corruption_type"]),
                    "epsilon": float(rec["epsilon"]),
                    "defense": defense,
                    "value": float(rec["value"]),
                    "score": float(rec["score"]),
                }
            )

    heuristic_auc: dict[str, float] = {}
    for heur in HEURISTICS:
        by_eps = []
        for eps in eps_grid:
            scores = [
                row["score"]
                for row in curves
                if row["defense"] == heur and float(row["epsilon"]) == float(eps)
            ]
            by_eps.append(_mean(scores))
        heuristic_auc[heur] = auc(by_eps, eps_grid)
    best_heuristic = max(heuristic_auc.items(), key=lambda kv: kv[1])[0]

    score_vs_eps = {"eps": eps_grid}
    for defense in ["no_defense", best_heuristic, "cachemedic"]:
        vals = []
        for eps in eps_grid:
            vals.append(
                _mean(
                    [
                        row["score"]
                        for row in curves
                        if row["defense"] == defense and float(row["epsilon"]) == float(eps)
                    ]
                )
            )
        key = "best_heuristic" if defense == best_heuristic else defense
        score_vs_eps[key] = [float(v) for v in vals]

    robustness_auc = {
        "no_defense": auc(score_vs_eps["no_defense"], eps_grid),
        "best_heuristic": auc(score_vs_eps["best_heuristic"], eps_grid),
        "cachemedic": auc(score_vs_eps["cachemedic"], eps_grid),
    }

    clean_regression = {}
    for defense in ["no_defense", best_heuristic, "cachemedic"]:
        vals = [_to_score(str(t["name"]), clean_metrics[str(t["name"])][defense]) for t in task_specs]
        key = "best_heuristic" if defense == best_heuristic else defense
        clean_regression[key] = _mean(vals)

    overhead = {}
    if bool(cfg["eval"]["measure_overhead"]):
        defense_map = {
            "no_defense": "no_defense",
            "best_heuristic": best_heuristic,
            "cachemedic": "cachemedic",
        }
        for out_key, src_key in defense_map.items():
            stat = overhead_raw[src_key]
            t_total = max(stat["time"], 1e-8)
            overhead[out_key] = {
                "tokens_per_sec": float(stat["tokens"] / t_total),
                "wall_time_sec": float(stat["time"]),
                "gpu_mem_gb": float(stat["gpu_mem_gb"]),
            }

    out = {
        "run_id": run_id,
        "config_hash": config_hash,
        "metrics": {
            "task_curves": curves,
            "clean_metrics": clean_metrics,
            "score_vs_eps": score_vs_eps,
            "best_heuristic_name": best_heuristic,
            "heuristic_auc": heuristic_auc,
            "robustness_auc": robustness_auc,
            "clean_regression": clean_regression,
            "overhead": overhead,
        },
        "settings": {
            "tasks": task_specs,
            "eps": eps_grid,
            "types": corr_types,
            "ood_protocol": cfg["eval"].get("ood_protocol"),
            "checkpoint_loaded": loaded_ckpt,
        },
    }
    write_json(run_dir / "metrics" / "task_metrics.json", out)

    eval_log_path = run_dir / "logs" / "eval.jsonl"
    for eps, v0, vh, vc in zip(
        eps_grid,
        score_vs_eps["no_defense"],
        score_vs_eps["best_heuristic"],
        score_vs_eps["cachemedic"],
    ):
        append_jsonl(
            eval_log_path,
            {
                "run_id": run_id,
                "phase": "eval",
                "timestamp_utc": utc_now_iso(),
                "step": int(round(eps * 1000)),
                "epsilon": eps,
                "best_heuristic": best_heuristic,
                "score_no_defense": v0,
                "score_best_heuristic": vh,
                "score_cachemedic": vc,
                "tokens_per_sec": overhead.get("cachemedic", {}).get("tokens_per_sec", 0.0),
                "gpu_mem_gb": overhead.get("cachemedic", {}).get("gpu_mem_gb", 0.0),
            },
        )
    elapsed_total = persist_progress(status="completed", reason="completed", current_unit=None)
    return {
        "run_id": run_id,
        "status": "completed",
        "best_heuristic": best_heuristic,
        "robustness_auc": robustness_auc,
        "elapsed_wall_sec": elapsed_total,
        "resumed": resumed_from is not None,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate CacheMedic++ robustness.")
    p.add_argument("--config", required=True, help="Path to YAML config.")
    p.add_argument("--run_dir", required=True, help="Output run directory.")
    p.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume eval from metrics/eval_progress.json if available (default).",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore eval progress state and start eval from scratch.",
    )
    p.add_argument(
        "--heartbeat_every",
        type=int,
        default=1,
        help="Emit eval heartbeat/progress after this many completed units (default: 1).",
    )
    p.set_defaults(resume=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_eval(
        Path(args.config).resolve(),
        Path(args.run_dir).resolve(),
        resume=bool(args.resume),
        heartbeat_every=int(args.heartbeat_every),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
