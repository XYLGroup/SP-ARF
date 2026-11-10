"""SP-ARF network and ablation levels 1-7."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import ModelConfig
from .eaf import EAF
from .sac import SAC
from .tdc import TDC

LEVELS = {

    1: (False, False, False, False, False, False),
    2: (True, False, False, False, False, False),
    3: (True, True, False, False, False, False),
    4: (True, True, True, False, False, False),
    5: (True, True, True, True, False, False),
    6: (True, True, True, True, True, False),
    7: (True, True, True, True, True, True),
}


class SPARF(nn.Module):
    def __init__(self, hrrp_dim: int, num_classes: int,
                 cfg: Optional[ModelConfig] = None, level: int = 7):
        super().__init__()
        if level not in LEVELS:
            raise ValueError(f"level must be in {sorted(LEVELS)}")
        cfg = cfg or ModelConfig()
        self.cfg = cfg
        self.level = level
        (self.use_tdc, self.use_sac, use_shift,
         use_cross, use_energy, use_gate) = LEVELS[level]

        self.tdc = TDC(cfg.d_z, cfg.d_c, cfg.d_e)
        self.sac = SAC(cfg.delta_max, cfg.rho_th, use_shift=use_shift)
        self.eaf = EAF(hrrp_dim, num_classes, cfg.d_model, cfg.n_heads, cfg.d_v,
                       cfg.dropout, use_cross, use_gate, use_energy)


    def complete(self, batch: Dict[str, torch.Tensor]):
        t, rcs_in, hrrp_in = batch["t"], batch["rcs_in"], batch["hrrp_in"]
        mask_r, mask_h = batch["mask_rcs"], batch["mask_hrrp"]

        if self.use_tdc:
            hat_raw, _ = self.tdc(t, rcs_in, mask_r, batch.get("valid"))
            hat_rcs = torch.where(mask_r > 0.5, rcs_in, hat_raw)
        else:
            hat_raw = self._linear_fill_1d(t, rcs_in, mask_r)
            hat_rcs = hat_raw

        skip_sac = batch.get("skip_sac", False)
        if torch.is_tensor(skip_sac):
            skip_sac = bool(skip_sac.reshape(-1)[0].item())
        if skip_sac:
            hat_hrrp = hrrp_in
        elif self.use_sac:
            hat_hrrp = self.sac(t, hrrp_in, mask_h)
            hat_hrrp = torch.where((mask_h > 0.5).unsqueeze(-1), hrrp_in, hat_hrrp)
        else:
            hat_hrrp = self._linear_fill_hrrp(t, hrrp_in, mask_h)
        return hat_rcs, hat_hrrp, hat_raw


    @staticmethod
    def _linear_fill_1d(t, y, mask):
        t_np, y_np = t.detach().cpu().numpy(), y.detach().cpu().numpy()
        m_np = (mask > 0.5).detach().cpu().numpy()
        filled = y_np.copy()
        for b in range(y.shape[0]):
            obs, miss = m_np[b], ~m_np[b]
            if obs.any() and miss.any():
                filled[b, miss] = np.interp(t_np[b, miss], t_np[b, obs], y_np[b, obs])
        out = torch.from_numpy(filled).to(device=y.device, dtype=y.dtype)
        return torch.where(mask > 0.5, y, out)

    def _linear_fill_hrrp(self, t, hrrp, mask):
        t_np, h_np = t.detach().cpu().numpy(), hrrp.detach().cpu().numpy()
        m_np = (mask > 0.5).detach().cpu().numpy()
        filled = h_np.copy()
        for b in range(hrrp.shape[0]):
            obs, miss = m_np[b], ~m_np[b]
            if not (obs.any() and miss.any()):
                continue
            for d in range(hrrp.shape[-1]):
                filled[b, miss, d] = np.interp(t_np[b, miss], t_np[b, obs],
                                               h_np[b, obs, d])
        out = torch.from_numpy(filled).to(device=hrrp.device, dtype=hrrp.dtype)
        m3 = (mask > 0.5).unsqueeze(-1).expand_as(hrrp)
        return torch.where(m3, hrrp, out)


    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        hat_rcs, hat_hrrp, hat_raw = self.complete(batch)
        return {"logits": self.eaf(hat_rcs, hat_hrrp),
                "hat_rcs": hat_rcs,
                "hat_rcs_raw": hat_raw,
                "hat_hrrp": hat_hrrp}


PhyARF = SPARF
