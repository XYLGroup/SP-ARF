"""Training objective."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

MODES = ("paper", "released")


def sp_arf_loss(outputs: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor],
                lambda_rec: float = 0.1,
                lambda_dyn: float = 0.1,
                mode: str = "paper") -> Dict[str, torch.Tensor]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    loss_cls = F.cross_entropy(outputs["logits"], batch["label"])

    gt = batch["rcs_gt"]
    mask = batch["mask_rcs"]
    valid = batch.get("valid", torch.ones_like(mask))
    sel = mask * valid

    if mode == "paper":
        hat = outputs["hat_rcs_raw"]
        loss_rec = (((hat - gt) ** 2) * sel).sum() / sel.sum().clamp_min(1.0)
        loss_dyn = hat.new_zeros(())
        total = loss_cls + lambda_rec * loss_rec
    else:
        hat = outputs["hat_rcs"]
        loss_rec = (((hat - gt) ** 2) * sel).sum() / sel.sum().clamp_min(1.0)
        dt = (batch["t"][:, 1:] - batch["t"][:, :-1]).clamp_min(1e-4)
        v1 = (hat[:, 1:] - hat[:, :-1]) / dt
        loss_dyn = (((v1[:, 1:] - v1[:, :-1]) ** 2).mean()
                    if hat.size(1) > 2 else hat.new_zeros(()))
        total = loss_cls + lambda_rec * loss_rec + lambda_dyn * loss_dyn

    return {"loss": total, "loss_cls": loss_cls.detach(),
            "loss_rec": loss_rec.detach(), "loss_dyn": loss_dyn.detach()}
