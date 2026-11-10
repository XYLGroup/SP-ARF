"""Train SP-ARF."""
#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sp_arf import DataConfig, ModelConfig, TrainConfig          # noqa: E402
from sp_arf.data import (WindowDataset, load_dataset,             # noqa: E402
                         split_trajectory, window_records)
from sp_arf.engine import evaluate, fit                           # noqa: E402
from sp_arf.models import SPARF                                   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--level", type=int, default=7, help="ablation level 1-7")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--loss", default="paper", choices=["paper", "released"],
                    help="paper = Eq. 39 of the manuscript (default); "
                         "released = the legacy objective that produced "
                         "checkpoints/SP-ARF.pth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="sp_arf")
    args = ap.parse_args()

    dcfg, cfg = DataConfig(), ModelConfig()
    out = Path(args.out) / f"{args.tag}_l{args.level}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"[data] loading {args.data}", flush=True)
    trajs = load_dataset(Path(args.data), pol=dcfg.pol)
    parts = split_trajectory(trajs, dcfg.train_traj_ids, dcfg.test_traj_ids)

    def loader(split: str, shuffle: bool) -> DataLoader:
        recs = window_records(parts[split], args.window, args.stride)
        return DataLoader(WindowDataset(parts[split], recs, args.window),
                          batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=args.device.startswith("cuda"))

    train_loader, test_loader = loader("train", True), loader("test", False)
    print(f"[data] train={len(train_loader.dataset)} test={len(test_loader.dataset)} "
          f"windows ({len(parts['train'])} / {len(parts['test'])} trajectories)",
          flush=True)

    model, info = fit(
        SPARF(dcfg.hrrp_dim, dcfg.num_classes, cfg, level=args.level),
        train_loader, args.epochs, args.lr, args.weight_decay, args.device,
        seed=args.seed, loss_mode=args.loss,
    )

    metrics, y_true, y_pred = evaluate(model, test_loader, args.device,
                                       return_predictions=True)
    torch.save(model.state_dict(), out / "model.pth")


    records = test_loader.dataset.records
    per_traj: dict = {}
    for (traj_idx, _start, _y), ok in zip(records, (y_true == y_pred).tolist()):
        per_traj.setdefault(traj_idx, [0, 0])
        per_traj[traj_idx][0] += int(ok)
        per_traj[traj_idx][1] += 1
    with open(out / "per_trajectory.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["trajectory_id", "run_id", "method", "correct", "total"])
        for traj_idx, (corr, tot) in sorted(per_traj.items()):
            tr = parts["test"][traj_idx]
            writer.writerow([f"{tr.label}_{tr.traj_id}", args.seed,
                             "SP-ARF", corr, tot])

    record = {"config": vars(args), "data": vars(dcfg), "model": vars(cfg),
              "train": info, "test_metrics": metrics,
              "n_test_windows": len(test_loader.dataset),
              "checkpoint": str(out / "model.pth")}
    with open(out / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    torch.save({"y_true": y_true, "y_pred": y_pred}, out / "predictions.pt")

    print(f"\n[test] Acc={metrics['Accuracy']:.4f}  P={metrics['Precision']:.4f}  "
          f"R={metrics['Recall']:.4f}  F1={metrics['F1']:.4f}  "
          f"Kappa={metrics['Kappa']:.4f}")
    print(f"[out]  {out}")


if __name__ == "__main__":
    main()
