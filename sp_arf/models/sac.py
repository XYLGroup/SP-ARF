"""Scattering-Aligned Completion (SAC)."""
from __future__ import annotations

import torch
import torch.nn as nn


def _shift_pad(x: torch.Tensor, delta: int) -> torch.Tensor:
    D = x.shape[-1]
    if delta == 0:
        return x
    out = torch.zeros_like(x)
    if delta > 0:
        out[delta:] = x[: D - delta]
    else:
        out[: D + delta] = x[-delta:]
    return out


def estimate_shift(a: torch.Tensor, b: torch.Tensor, delta_max: int) -> int:
    best_delta, best_score = 0, -1e9
    nb = torch.norm(b) + 1e-8
    for delta in range(-delta_max, delta_max + 1):
        sa = _shift_pad(a, delta)
        score = torch.dot(sa, b) / (torch.norm(sa) * nb + 1e-8)
        if score > best_score:
            best_score, best_delta = float(score), delta
    return best_delta


class SAC(nn.Module):
    def __init__(self, delta_max: int = 15, rho_th: float = 0.7,
                 use_shift: bool = True):
        super().__init__()
        self.delta_max = delta_max
        self.rho_th = rho_th
        self.use_shift = use_shift

    @torch.no_grad()
    def forward(self, t, hrrp_in, mask_hrrp):
        B, L, D = hrrp_in.shape
        out = hrrp_in.clone()
        for b in range(B):
            obs = torch.where(mask_hrrp[b] > 0.5)[0].tolist()
            if not obs:
                continue
            for ell in range(L):
                if mask_hrrp[b, ell] > 0.5:
                    continue
                left = [j for j in obs if j < ell]
                right = [j for j in obs if j > ell]
                if left and right:
                    a, bb = left[-1], right[0]
                elif left:
                    a = bb = left[-1]
                else:
                    a = bb = right[0]

                sa, sb = out[b, a], out[b, bb]
                if a == bb:
                    out[b, ell] = sa
                    continue

                if self.use_shift:
                    delta = estimate_shift(sa, sb, self.delta_max)
                    sa_al = _shift_pad(sa, delta)
                else:
                    delta, sa_al = 0, sa
                rho = torch.dot(sa_al, sb) / (torch.norm(sa_al) * torch.norm(sb) + 1e-8)
                alpha = float((t[b, ell] - t[b, a]) / (t[b, bb] - t[b, a] + 1e-6))
                alpha = max(0.0, min(1.0, alpha))

                if float(rho) >= self.rho_th:
                    d1 = int(round(alpha * delta))
                    d2 = int(round(-(1.0 - alpha) * delta))
                    out[b, ell] = ((1 - alpha) * _shift_pad(sa, d1)
                                   + alpha * _shift_pad(sb, d2))
                else:
                    out[b, ell] = sa if alpha < 0.5 else sb
        return out
