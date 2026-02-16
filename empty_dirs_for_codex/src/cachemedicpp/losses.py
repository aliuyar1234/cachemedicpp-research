"""Losses for CacheMedic++ training objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


EPS0 = 1e-8


def kd_loss(z_clean: torch.Tensor, z_rep: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    T = float(temperature)
    p = F.softmax(z_clean / T, dim=-1)
    log_q = F.log_softmax(z_rep / T, dim=-1)
    return F.kl_div(log_q, p, reduction="batchmean")


def identity_loss(
    K_clean: torch.Tensor,
    V_clean: torch.Tensor,
    K_rep: torch.Tensor,
    V_rep: torch.Tensor,
    eps0: float = EPS0,
) -> torch.Tensor:
    num = (K_rep - K_clean).pow(2).sum() + (V_rep - V_clean).pow(2).sum()
    den = K_clean.pow(2).sum() + V_clean.pow(2).sum() + float(eps0)
    return num / den


def contraction_ratio(
    K_clean: torch.Tensor,
    V_clean: torch.Tensor,
    K_corr: torch.Tensor,
    V_corr: torch.Tensor,
    K_rep: torch.Tensor,
    V_rep: torch.Tensor,
    apply_to: str = "KV",
    eps0: float = EPS0,
) -> torch.Tensor:
    mode = str(apply_to).upper()
    use_k = "K" in mode
    use_v = "V" in mode
    if not (use_k or use_v):
        use_k = True
        use_v = True

    num = torch.zeros((), device=K_clean.device, dtype=K_clean.dtype)
    den = torch.zeros((), device=K_clean.device, dtype=K_clean.dtype)
    if use_k:
        deltaK = K_corr - K_clean
        deltaK_rep = K_rep - K_clean
        num = num + torch.norm(deltaK_rep)
        den = den + torch.norm(deltaK)
    if use_v:
        deltaV = V_corr - V_clean
        deltaV_rep = V_rep - V_clean
        num = num + torch.norm(deltaV_rep)
        den = den + torch.norm(deltaV)
    den = den + float(eps0)
    return num / den


def contraction_hinge_loss(rho: torch.Tensor, alpha_contr: float) -> torch.Tensor:
    return torch.relu(rho - float(alpha_contr)).pow(2)


@dataclass
class LayerTriplet:
    layer: int
    K_clean: torch.Tensor
    V_clean: torch.Tensor
    K_corr: torch.Tensor
    V_corr: torch.Tensor
    K_rep: torch.Tensor
    V_rep: torch.Tensor


def aggregate_objective(
    *,
    z_clean: torch.Tensor,
    z_rep: torch.Tensor,
    layer_triplets: list[LayerTriplet],
    temperature: float,
    lambda_id: float,
    lambda_contr: float,
    alpha_contr: float,
    corruption_active: bool,
    apply_to: str = "KV",
    lambda_w: float = 0.0,
    repair_parameters: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    loss_kd = kd_loss(z_clean, z_rep, temperature=temperature)

    id_terms = []
    rho_values = []
    contr_terms = []
    for trip in layer_triplets:
        id_terms.append(identity_loss(trip.K_clean, trip.V_clean, trip.K_rep, trip.V_rep))
        rho = contraction_ratio(
            trip.K_clean,
            trip.V_clean,
            trip.K_corr,
            trip.V_corr,
            trip.K_rep,
            trip.V_rep,
            apply_to=apply_to,
        )
        rho_values.append(rho)
        contr_terms.append(contraction_hinge_loss(rho, alpha_contr))

    if id_terms:
        loss_id = torch.stack(id_terms).sum() if not corruption_active else torch.zeros_like(loss_kd)
    else:
        loss_id = torch.zeros_like(loss_kd)
    if contr_terms and corruption_active:
        loss_contr = torch.stack(contr_terms).sum()
    else:
        loss_contr = torch.zeros_like(loss_kd)

    loss_w = torch.zeros_like(loss_kd)
    if repair_parameters and float(lambda_w) > 0.0:
        norm_sq = torch.zeros_like(loss_kd)
        for p in repair_parameters:
            norm_sq = norm_sq + p.pow(2).sum()
        loss_w = float(lambda_w) * norm_sq

    total = loss_kd + float(lambda_id) * loss_id + float(lambda_contr) * loss_contr + loss_w

    if rho_values:
        rhos = torch.stack(rho_values).float()
        rho_mean = float(rhos.mean().item())
        rho_p95 = float(torch.quantile(rhos, 0.95).item())
    else:
        rho_mean = 0.0
        rho_p95 = 0.0

    return {
        "loss_total": total,
        "loss_kd": loss_kd,
        "loss_id": loss_id,
        "loss_contr": loss_contr,
        "rho_mean": rho_mean,
        "rho_p95": rho_p95,
    }
