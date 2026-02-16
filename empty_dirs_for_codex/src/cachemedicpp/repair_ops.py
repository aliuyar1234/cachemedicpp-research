"""Repair operator families A/B/C."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class _GatingMLP(nn.Module):
    def __init__(self, d_model: int, rank: int, alpha_max: float = 0.999) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, rank),
            nn.SiLU(),
            nn.Linear(rank, rank),
        )
        self.alpha_max = float(alpha_max)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        # q: [1, H, d]
        alpha = torch.sigmoid(self.net(q)) * self.alpha_max
        return alpha


class OptionARepair(nn.Module):
    """Query-conditioned low-rank additive correction."""

    def __init__(self, *, head_dim: int, rank: int, apply_to: str = "KV", alpha_max: float = 0.999) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.apply_to = str(apply_to)

        self.Wk = nn.Parameter(torch.empty((self.head_dim, self.rank)))
        self.Uk = nn.Parameter(torch.empty((self.head_dim, self.rank)))
        self.Wv = nn.Parameter(torch.empty((self.head_dim, self.rank)))
        self.Uv = nn.Parameter(torch.empty((self.head_dim, self.rank)))
        self.gate_k = _GatingMLP(self.head_dim, self.rank, alpha_max=alpha_max)
        self.gate_v = _GatingMLP(self.head_dim, self.rank, alpha_max=alpha_max)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in [self.Wk, self.Uk, self.Wv, self.Uv]:
            nn.init.xavier_uniform_(p)

    def _reshape_q(self, q: torch.Tensor) -> torch.Tensor:
        if q.ndim == 4:
            # [1, H, 1, d] -> [1, H, d]
            return q[:, :, -1, :]
        if q.ndim == 3:
            return q
        raise ValueError(f"Unexpected q shape for repair: {tuple(q.shape)}")

    def forward(
        self,
        q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qh = self._reshape_q(q)
        alpha_k = self.gate_k(qh).unsqueeze(2)  # [1,H,1,r]
        alpha_v = self.gate_v(qh).unsqueeze(2)

        K_hat = K
        V_hat = V

        if "K" in self.apply_to:
            Ck = K @ self.Wk  # [1,H,T,r]
            dK = (Ck * alpha_k) @ self.Uk.transpose(0, 1)  # [1,H,T,d]
            K_hat = K + dK
        if "V" in self.apply_to:
            Cv = V @ self.Wv
            dV = (Cv * alpha_v) @ self.Uv.transpose(0, 1)
            V_hat = V + dV
        return K_hat, V_hat


class OptionBRepair(nn.Module):
    """Learned-basis shrinkage with bounded scaling."""

    def __init__(self, *, head_dim: int, rank: int, apply_to: str = "KV", alpha_max: float = 0.9) -> None:
        super().__init__()
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.apply_to = str(apply_to)

        self.Bk = nn.Parameter(torch.empty((self.head_dim, self.rank)))
        self.Bv = nn.Parameter(torch.empty((self.head_dim, self.rank)))
        self.gate_k = _GatingMLP(self.head_dim, self.rank, alpha_max=alpha_max)
        self.gate_v = _GatingMLP(self.head_dim, self.rank, alpha_max=alpha_max)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.Bk)
        nn.init.xavier_uniform_(self.Bv)

    def _reshape_q(self, q: torch.Tensor) -> torch.Tensor:
        return q[:, :, -1, :] if q.ndim == 4 else q

    def forward(
        self,
        q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qh = self._reshape_q(q)
        sK = self.gate_k(qh).unsqueeze(2)
        sV = self.gate_v(qh).unsqueeze(2)

        K_hat = K
        V_hat = V
        if "K" in self.apply_to:
            ck = K @ self.Bk
            ck_shrunk = ck * sK
            K_hat = K - (ck - ck_shrunk) @ self.Bk.transpose(0, 1)
        if "V" in self.apply_to:
            cv = V @ self.Bv
            cv_shrunk = cv * sV
            V_hat = V - (cv - cv_shrunk) @ self.Bv.transpose(0, 1)
        return K_hat, V_hat


class OptionCRepair(nn.Module):
    """Shared-operator style repair with lightweight conditioning."""

    def __init__(
        self,
        *,
        head_dim: int,
        rank: int,
        num_heads: int,
        num_layers: int,
        layer_index: int,
        apply_to: str = "KV",
        alpha_max: float = 0.999,
        use_layer_embedding: bool = True,
        use_head_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.base = OptionARepair(head_dim=head_dim, rank=rank, apply_to=apply_to, alpha_max=alpha_max)
        self.use_layer_embedding = bool(use_layer_embedding)
        self.use_head_embedding = bool(use_head_embedding)
        self.layer_index = int(layer_index)
        if self.use_layer_embedding:
            self.layer_embed = nn.Embedding(num_layers, head_dim)
        else:
            self.layer_embed = None
        if self.use_head_embedding:
            self.head_embed = nn.Embedding(num_heads, head_dim)
        else:
            self.head_embed = None

    def forward(self, q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim == 4:
            qh = q[:, :, -1, :]
            needs_expand = True
        else:
            qh = q
            needs_expand = False

        q_cond = qh
        if self.layer_embed is not None:
            lid = torch.tensor([self.layer_index], device=qh.device, dtype=torch.long)
            q_cond = q_cond + self.layer_embed(lid).view(1, 1, -1)
        if self.head_embed is not None:
            h_ids = torch.arange(qh.shape[1], device=qh.device, dtype=torch.long)
            q_cond = q_cond + self.head_embed(h_ids).view(1, qh.shape[1], -1)

        if needs_expand:
            q_forward = q_cond.unsqueeze(2)
        else:
            q_forward = q_cond
        return self.base(q_forward, K, V)


def build_repair_operator(
    repair_cfg: dict[str, Any],
    *,
    head_dim: int,
    num_heads: int,
    layer_index: int,
    num_layers: int,
) -> nn.Module:
    family = str(repair_cfg.get("family", "A"))
    rank = int(repair_cfg.get("rank", 8))
    apply_to = str(repair_cfg.get("apply_to", "KV"))
    alpha_max = float(repair_cfg.get("alpha_max", 0.999))
    if family == "A":
        return OptionARepair(head_dim=head_dim, rank=rank, apply_to=apply_to, alpha_max=alpha_max)
    if family == "B":
        return OptionBRepair(head_dim=head_dim, rank=rank, apply_to=apply_to, alpha_max=alpha_max)
    if family == "C":
        cond = repair_cfg.get("conditioning") or {}
        return OptionCRepair(
            head_dim=head_dim,
            rank=rank,
            num_heads=num_heads,
            num_layers=num_layers,
            layer_index=layer_index,
            apply_to=apply_to,
            alpha_max=alpha_max,
            use_layer_embedding=bool(cond.get("use_layer_embedding", True)),
            use_head_embedding=bool(cond.get("use_head_embedding", True)),
        )
    raise ValueError(f"Unsupported repair family: {family}")

