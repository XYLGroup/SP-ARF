"""Evaluate a checkpoint."""
#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sp_arf import DataConfig, ModelConfig                        # noqa: E402
from sp_arf.data import (WindowDataset, load_dataset,              # noqa: E402
                         split_trajectory, window_records)
from sp_arf.engine import evaluate                                 # noqa: E402
from sp_arf.models import SPARF                                    # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "SP-ARF.pth"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "eval.json"))
    ap.add_argument("--level", type=int, default=7)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    dcfg, cfg = DataConfig(), ModelConfig()
    trajs = load_dataset(Path(args.data), pol=dcfg.pol)
    parts = split_trajectory(trajs, dcfg.train_traj_ids, dcfg.test_traj_ids)
    recs = window_records(parts["test"], args.window, args.stride)
    loader = DataLoader(WindowDataset(parts["test"], recs, args.window),
                        batch_size=args.batch_size, shuffle=False)

    model = SPARF(dcfg.hrrp_dim, dcfg.num_classes, cfg, level=args.level)
    state = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)


    stray = [k for k in unexpected if not k.startswith("eaf.cls_concat")]
    if missing or stray:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={stray}")
    model.to(args.device)

    metrics, y_true, y_pred = evaluate(model, loader, args.device, return_predictions=True)
    payload = {"checkpoint": str(args.ckpt), "level": args.level,
               "n_windows": len(loader.dataset), "metrics": metrics}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    torch.save({"y_true": y_true, "y_pred": y_pred}, out.with_suffix(".pt"))

    print(json.dumps(metrics, indent=2))
    print(f"[out] {out}")


if __name__ == "__main__":
    main()
