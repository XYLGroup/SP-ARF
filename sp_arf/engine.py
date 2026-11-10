"""Training and evaluation."""
from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .metrics import compute_metrics
from .models.loss import sp_arf_loss


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def _to_device(batch: Dict, device: str) -> Dict:
    if device == "cpu":
        return batch
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def train_epoch(model: nn.Module, loader: DataLoader, opt, sched, device: str,
                lambda_rec: float = 0.1, lambda_dyn: float = 0.1,
                loss_mode: str = "paper",
                grad_clip: float = 1.0) -> Dict[str, float]:
    model.train()
    total = rec = dyn = 0.0
    for n, batch in enumerate(loader, start=1):
        opt.zero_grad(set_to_none=True)
        batch = _to_device(batch, device)
        parts = sp_arf_loss(model(batch), batch, lambda_rec, lambda_dyn,
                            mode=loss_mode)
        parts["loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total += float(parts["loss"].detach())
        rec += float(parts["loss_rec"])
        dyn += float(parts["loss_dyn"])
    sched.step()
    n = max(n, 1)
    return {"loss": total / n, "loss_rec": rec / n, "loss_dyn": dyn / n}


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: str = "cuda",
             return_predictions: bool = False):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        ys.append(batch["label"].cpu().numpy())
        ps.append(model(batch)["logits"].argmax(-1).cpu().numpy())
    y_true, y_pred = np.concatenate(ys), np.concatenate(ps)
    metrics = compute_metrics(y_true, y_pred)
    return (metrics, y_true, y_pred) if return_predictions else metrics


def fit(model: nn.Module, loader: DataLoader, epochs: int, lr: float,
        weight_decay: float, device: str, seed: int = 42,
        lambda_rec: float = 0.1, lambda_dyn: float = 0.1,
        loss_mode: str = "paper",
        verbose: bool = True) -> Tuple[nn.Module, Dict]:
    set_seed(seed)
    model = model.to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(epochs, 1))

    t0, history = time.perf_counter(), []
    for ep in range(epochs):
        stats = train_epoch(model, loader, opt, sched, device, lambda_rec,
                            lambda_dyn, loss_mode)
        stats["epoch"] = ep + 1
        history.append(stats)
        if verbose:
            print(f"  epoch {ep + 1:2d}/{epochs}  loss={stats['loss']:.4f}  "
                  f"rec={stats['loss_rec']:.3e}  dyn={stats['loss_dyn']:.3e}",
                  flush=True)
    return model, {"epochs": epochs, "history": history, "loss_mode": loss_mode,
                   "train_time_s": round(time.perf_counter() - t0, 1)}
