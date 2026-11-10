"""Energy-Guided Adaptive Fusion (EAF)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class EnergyGuidedCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_energy: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_energy = use_energy
        self.w_qr = nn.Linear(d_model, d_model)
        self.w_kh = nn.Linear(d_model, d_model)
        self.w_vh = nn.Linear(d_model, d_model)
        self.w_qh = nn.Linear(d_model, d_model)
        self.w_kr = nn.Linear(d_model, d_model)
        self.w_vr = nn.Linear(d_model, d_model)
        self.proj_r = nn.Linear(d_model, d_model)
        self.proj_h = nn.Linear(d_model, d_model)
        self.lambda_e = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)
        self.norm_r = nn.LayerNorm(d_model)
        self.norm_h = nn.LayerNorm(d_model)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, W, _ = x.shape
        return x.view(B, W, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, e_r, e_h, hrrp):
        B, W, _ = e_r.shape


        q_r = self._heads(self.w_qr(e_r))
        k_h = self._heads(self.w_kh(e_h))
        v_h = self._heads(self.w_vh(e_h))
        scores_r = torch.matmul(q_r, k_h.transpose(-2, -1)) / math.sqrt(self.d_k)
        if self.use_energy:
            energy = (hrrp ** 2).sum(dim=-1)
            m_e = energy / (energy.max(dim=-1, keepdim=True).values + 1e-8)


            lam = torch.clamp_min(self.lambda_e, 0.0)
            scores_r = scores_r + lam * m_e.unsqueeze(1).expand(-1, W, -1).unsqueeze(1)
        a_r = torch.matmul(self.dropout(torch.softmax(scores_r, dim=-1)), v_h)
        out_r = self.norm_r(e_r + self.proj_r(
            a_r.transpose(1, 2).contiguous().view(B, W, -1)))


        q_h = self._heads(self.w_qh(e_h))
        k_r = self._heads(self.w_kr(e_r))
        v_r = self._heads(self.w_vr(e_r))
        scores_h = torch.matmul(q_h, k_r.transpose(-2, -1)) / math.sqrt(self.d_k)
        a_h = torch.matmul(self.dropout(torch.softmax(scores_h, dim=-1)), v_r)
        out_h = self.norm_h(e_h + self.proj_h(
            a_h.transpose(1, 2).contiguous().view(B, W, -1)))
        return out_r, out_h


class MultiScaleConvPool(nn.Module):
    def __init__(self, d_model: int, d_v: int):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv1d(d_model, d_model // 2, k, padding=k // 2) for k in (3, 5, 7)]
        )
        self.proj = nn.Linear(3 * (d_model // 2), d_v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        feats = [F.gelu(b(h)).mean(dim=-1) for b in self.branches]
        return self.proj(torch.cat(feats, dim=-1))


class EAF(nn.Module):
    def __init__(self, hrrp_dim: int, num_classes: int, d_model: int = 128,
                 n_heads: int = 8, d_v: int = 128, dropout: float = 0.1,
                 use_cross_attn: bool = True, use_gate: bool = True,
                 use_energy: bool = True):
        super().__init__()
        self.use_cross_attn = use_cross_attn
        self.use_gate = use_gate
        self.proj_r = nn.Linear(1, d_model)
        self.proj_h = nn.Linear(hrrp_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.cross = EnergyGuidedCrossAttention(d_model, n_heads, dropout,
                                                use_energy=use_energy)
        self.pool_r = MultiScaleConvPool(d_model, d_v)
        self.pool_h = MultiScaleConvPool(d_model, d_v)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * d_v, d_v), nn.ReLU(), nn.Linear(d_v, 2)
        )
        self.cls = nn.Sequential(
            nn.Linear(d_v, d_v), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_v, num_classes),
        )


        self.cls_concat = nn.Sequential(
            nn.Linear(2 * d_v, d_v), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_v, num_classes),
        )

    def forward(self, rcs: torch.Tensor, hrrp: torch.Tensor) -> torch.Tensor:
        e_r = self.pos(self.proj_r(rcs.unsqueeze(-1)))
        e_h = self.pos(self.proj_h(hrrp))
        if self.use_cross_attn:
            e_r, e_h = self.cross(e_r, e_h, hrrp)
        v_r = self.pool_r(e_r)
        v_h = self.pool_h(e_h)
        if self.use_gate:
            w = torch.softmax(self.gate_mlp(torch.cat([v_r, v_h], dim=-1)), dim=-1)
            return self.cls(w[:, 0:1] * v_r + w[:, 1:2] * v_h)

        return self.cls(0.5 * (v_r + v_h))
