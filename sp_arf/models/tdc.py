"""Temporal-Dynamics Completion (TDC)."""
from __future__ import annotations

import torch
import torch.nn as nn


def rk2_step(f, z, dt, c):
    k1 = f(z, c)
    k2 = f(z + dt * k1, c)
    return z + 0.5 * dt * (k1 + k2)


class ODEFunc(nn.Module):
    def __init__(self, d_z: int, d_c: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_z + d_c, d_z), nn.Tanh(), nn.Linear(d_z, d_z)
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, c], dim=-1))


class TDC(nn.Module):
    def __init__(self, d_z: int = 64, d_c: int = 64, d_e: int = 64):
        super().__init__()
        self.d_z = d_z
        self.emb = nn.Sequential(nn.Linear(1, d_e), nn.Tanh(), nn.Linear(d_e, d_e))
        self.map = nn.Sequential(
            nn.Linear(d_z + 2, d_c), nn.Tanh(), nn.Linear(d_c, d_c)
        )
        self.ode = ODEFunc(d_z, d_c)
        self.W_g = nn.Linear(d_e, d_z)
        self.U_g = nn.Linear(d_z, d_z, bias=False)
        self.W_q = nn.Linear(d_e, d_z)
        self.U_q = nn.Linear(d_z, d_z, bias=False)
        self.W_z = nn.Linear(d_e, d_z)
        self.U_z = nn.Linear(d_z, d_z, bias=False)
        self.dec = nn.Sequential(nn.Linear(d_z, d_z), nn.Tanh(), nn.Linear(d_z, 1))
        self.z0 = nn.Parameter(torch.zeros(d_z))

    def forward(self, t, rcs_in, mask_rcs, valid=None):
        B, L = rcs_in.shape
        z = self.z0.unsqueeze(0).expand(B, -1).contiguous()
        hats, prev_r, prev_prev_r = [], z.new_zeros(B), z.new_zeros(B)

        for ell in range(L):
            if ell == 0:
                dt = z.new_full((B,), 1e-3)
            else:
                dt = (t[:, ell] - t[:, ell - 1]).clamp_min(1e-4)


            if ell <= 1:
                nu = z.new_zeros(B)
            else:
                dt_prev = (t[:, ell - 1] - t[:, ell - 2]).clamp_min(1e-4)
                nu = (prev_r - prev_prev_r) / (dt_prev + 1e-6)

            c = self.map(torch.cat([z, nu.unsqueeze(-1), dt.unsqueeze(-1)], dim=-1))
            z_tilde = rk2_step(self.ode, z, dt.unsqueeze(-1), c)

            e = self.emb(rcs_in[:, ell: ell + 1])
            g = torch.sigmoid(self.W_g(e) + self.U_g(z_tilde))
            q = torch.sigmoid(self.W_q(e) + self.U_q(z_tilde))
            z_bar = torch.tanh(self.W_z(e) + self.U_z(q * z_tilde))
            z_new = g * z_tilde + (1.0 - g) * z_bar
            z = torch.where((mask_rcs[:, ell] > 0.5).unsqueeze(-1), z_new, z_tilde)

            r_hat = self.dec(z).squeeze(-1)
            if valid is not None:
                r_hat = r_hat * valid[:, ell]
            hats.append(r_hat)
            prev_prev_r, prev_r = prev_r, r_hat

        return torch.stack(hats, dim=1), z.unsqueeze(1)
