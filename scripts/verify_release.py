"""Verify the released weights."""
#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sp_arf import DataConfig, ModelConfig          # noqa: E402
from sp_arf.data import (WindowDataset, load_dataset,  # noqa: E402
                         split_trajectory, window_records)
from sp_arf.engine import evaluate                  # noqa: E402
from sp_arf.models import SPARF                     # noqa: E402

EXPECTED_ACC = 0.8911


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "SP-ARF.pth"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--stride", type=int, default=20)
    args = ap.parse_args()

    dcfg, cfg = DataConfig(), ModelConfig()
    trajs = load_dataset(Path(args.data), pol=dcfg.pol)
    parts = split_trajectory(trajs, dcfg.train_traj_ids, dcfg.test_traj_ids)
    recs = window_records(parts["test"], args.window, args.stride)
    loader = DataLoader(WindowDataset(parts["test"], recs, args.window),
                        batch_size=64, shuffle=False)

    model = SPARF(dcfg.hrrp_dim, dcfg.num_classes, cfg, level=7)
    state = torch.load(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)


    stray = [k for k in unexpected if not k.startswith("eaf.cls_concat")]
    if missing or stray:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={stray}")
    model.to(args.device)

    m = evaluate(model, loader, args.device)
    print(f"test windows : {len(loader.dataset)}")
    print(f"Accuracy     : {m['Accuracy']:.4f}   (expected {EXPECTED_ACC:.4f})")
    print(f"Precision    : {m['Precision']:.4f}")
    print(f"Recall       : {m['Recall']:.4f}")
    print(f"F1           : {m['F1']:.4f}")
    print(f"Kappa        : {m['Kappa']:.4f}")
    ok = abs(m["Accuracy"] - EXPECTED_ACC) < 0.005
    print("RESULT       :", "OK" if ok else "MISMATCH - check the data pipeline")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
