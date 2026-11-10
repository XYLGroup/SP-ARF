"""Data loading, trajectory split, completion and windowing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ("EA-18G", "EP-3E", "F2", "F15", "F16", "F18", "F22", "F35", "IDF")
PAPER_CLASS_NAMES = ("EA-18G", "EP-3E", "F-2", "F-15", "F-16", "F-18", "F-22",
                     "F-35", "IDF")
NUM_TRAJ = 5
_CLASS_INDEX = {c.replace("-", "").upper(): i for i, c in enumerate(CLASS_NAMES)}


def class_index(name: str) -> int:
    key = str(name).replace("-", "").upper()
    if key not in _CLASS_INDEX:
        raise KeyError(f"unknown target class {name!r}")
    return _CLASS_INDEX[key]


@dataclass
class Trajectory:
    rcs: np.ndarray
    t_rcs: np.ndarray
    hrrp: np.ndarray
    t_hrrp: np.ndarray
    gt_rcs: np.ndarray
    gt_hrrp: np.ndarray
    t_u: np.ndarray
    mask_rcs: np.ndarray
    mask_hrrp: np.ndarray
    label: int
    traj_id: int

    @property
    def length(self) -> int:
        return len(self.t_u)


def _load_npy_dict(path: Path) -> dict:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and (obj.shape == () or obj.dtype == object):
        return obj.item()
    raise TypeError(f"unexpected npy payload in {path}")


def load_trajectory(async_path: Path, sync_path: Optional[Path] = None,
                    pol: str = "CoHH") -> Trajectory:
    raw = _load_npy_dict(async_path)
    t_rcs = np.asarray(raw["rcs_time"], dtype=np.int64)
    t_hrrp = np.asarray(raw["hrrp_time"], dtype=np.int64)
    rcs = np.asarray(raw["rcs_data"][pol], dtype=np.float32)
    hrrp = np.asarray(raw["hrrp_data"][pol], dtype=np.float32)
    L = int(raw.get("total_steps", max(t_rcs.max(), t_hrrp.max()) + 1))

    if np.intersect1d(t_rcs, t_hrrp).size:
        raise ValueError(f"{async_path.name}: timestamps are not strictly asynchronous")

    mask_rcs, mask_hrrp = np.zeros(L, bool), np.zeros(L, bool)
    mask_rcs[t_rcs], mask_hrrp[t_hrrp] = True, True

    if sync_path is not None and Path(sync_path).exists():
        sync = _load_npy_dict(sync_path)
        gt_rcs = np.asarray(sync["rcs_data"][pol], dtype=np.float32)
        gt_hrrp = np.asarray(sync["hrrp_data"][pol], dtype=np.float32)
    else:
        gt_rcs = np.zeros(L, np.float32)
        gt_hrrp = np.zeros((L, hrrp.shape[1]), np.float32)
        gt_rcs[t_rcs], gt_hrrp[t_hrrp] = rcs, hrrp

    return Trajectory(
        rcs=rcs, t_rcs=t_rcs.astype(np.float32),
        hrrp=hrrp, t_hrrp=t_hrrp.astype(np.float32),
        gt_rcs=gt_rcs, gt_hrrp=gt_hrrp,
        t_u=np.arange(L, dtype=np.float32),
        mask_rcs=mask_rcs, mask_hrrp=mask_hrrp,
        label=class_index(raw["target_type"]),
        traj_id=int(async_path.stem.split("_traj_")[1].split("_")[0]),
    )


def load_dataset(root: Path, pol: str = "CoHH") -> List[Trajectory]:
    root, out = Path(root), []
    for ci, cls in enumerate(CLASS_NAMES):
        for tid in range(1, NUM_TRAJ + 1):
            ap = root / f"{cls}_traj_{tid}_async.npy"
            if not ap.exists():
                raise FileNotFoundError(
                    f"{ap} not found. The dataset is distributed separately; see README.")
            tr = load_trajectory(ap, root / f"{cls}_traj_{tid}_sync.npy", pol)
            tr.label, tr.traj_id = ci, tid
            out.append(tr)
    return out


def split_trajectory(trajs: Sequence[Trajectory], train_ids=(1, 2, 3),
                     test_ids=(4, 5)) -> Dict[str, List[Trajectory]]:
    return {"train": [t for t in trajs if t.traj_id in train_ids],
            "test": [t for t in trajs if t.traj_id in test_ids]}


def _extrapolate_edges(t: np.ndarray, y: np.ndarray, obs: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(obs)
    if idx.size < 2:
        return y
    out = y.copy()
    first, second = idx[0], idx[1]
    left = np.arange(first)
    if left.size:
        out[left] = y[first] + (y[second] - y[first]) / (t[second] - t[first]) \
            * (t[left] - t[first])
    last, prev = idx[-1], idx[-2]
    right = np.arange(last + 1, t.size)
    if right.size:
        out[right] = y[last] + (y[last] - y[prev]) / (t[last] - t[prev]) \
            * (t[right] - t[last])
    return out


def _fill_1d(t, y, obs):
    out = np.zeros_like(y, dtype=np.float64)
    out[obs] = y[obs]
    if obs.sum() >= 2:
        out = _extrapolate_edges(t, out, obs)
        out[~obs] = np.interp(t[~obs], t[obs], y[obs])
    elif obs.sum() == 1:
        out[~obs] = y[obs][0]
    return out.astype(np.float32)


def complete_linear(tr: Trajectory) -> Tuple[np.ndarray, np.ndarray]:
    rcs = np.zeros(tr.length, np.float32)
    rcs[tr.mask_rcs] = tr.rcs
    rcs = _fill_1d(tr.t_u, rcs, tr.mask_rcs)

    hrrp = np.zeros((tr.length, tr.hrrp.shape[1]), np.float32)
    hrrp[tr.mask_hrrp] = tr.hrrp
    for d in range(hrrp.shape[1]):
        hrrp[:, d] = _fill_1d(tr.t_u, hrrp[:, d], tr.mask_hrrp)
    return rcs, hrrp


def _shift_pad_np(x: np.ndarray, delta: int) -> np.ndarray:
    D = x.shape[-1]
    if delta == 0:
        return x
    out = np.zeros_like(x)
    if delta > 0:
        out[delta:] = x[: D - delta]
    else:
        out[: D + delta] = x[-delta:]
    return out


def sac_complete_trajectory(t, hrrp, mask, delta_max: int = 15,
                            rho_th: float = 0.7) -> np.ndarray:
    out = hrrp.astype(np.float32).copy()
    obs = np.flatnonzero(mask > 0.5)
    if obs.size == 0:
        return out
    for ell in np.flatnonzero(mask <= 0.5):
        left, right = obs[obs < ell], obs[obs > ell]
        if left.size and right.size:
            a, b = int(left[-1]), int(right[0])
        elif left.size:
            a = b = int(left[-1])
        else:
            a = b = int(right[0])
        sa, sb = out[a], out[b]
        if a == b:
            out[ell] = sa
            continue

        best_d, best_s = 0, -1e9
        nb = float(np.linalg.norm(sb)) + 1e-8
        for d in range(-delta_max, delta_max + 1):
            shifted = _shift_pad_np(sa, d)
            s = float(np.dot(shifted, sb) / (np.linalg.norm(shifted) * nb + 1e-8))
            if s > best_s:
                best_s, best_d = s, d
        delta = best_d
        alpha = float((t[ell] - t[a]) / (t[b] - t[a] + 1e-6))
        alpha = max(0.0, min(1.0, alpha))
        if best_s >= rho_th:
            d1 = int(round(alpha * delta))
            d2 = int(round(-(1.0 - alpha) * delta))
            out[ell] = (1.0 - alpha) * _shift_pad_np(sa, d1) \
                + alpha * _shift_pad_np(sb, d2)
        else:
            out[ell] = sa if alpha < 0.5 else sb
    return out


def complete_sac(tr: Trajectory, delta_max: int = 15,
                 rho_th: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    rcs, _ = complete_linear(tr)
    h = np.zeros_like(tr.gt_hrrp, dtype=np.float32)
    h[tr.mask_hrrp] = tr.hrrp
    return rcs, sac_complete_trajectory(tr.t_u, h, tr.mask_hrrp, delta_max, rho_th)


def n_windows(length: int, W: int, S: int) -> int:
    return max(0, (length - W) // S + 1)


def window_records(trajs: Sequence[Trajectory], W: int, S: int
                   ) -> List[Tuple[int, int, int]]:
    return [(i, st, t.label)
            for i, t in enumerate(trajs)
            for st in range(0, n_windows(t.length, W, S) * S, S)]


class WindowDataset(Dataset):
    def __init__(self, trajs: Sequence[Trajectory], records, W: int,
                 completer=complete_sac):
        self.trajs, self.records, self.W = list(trajs), list(records), W
        self.comp = [completer(t) for t in self.trajs]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        i, st, y = self.records[idx]
        tr, (rcs, hrrp) = self.trajs[i], self.comp[i]
        sl = slice(st, st + self.W)
        mask_r = tr.mask_rcs[sl].astype(np.float32)
        return {
            "t": torch.from_numpy(tr.t_u[sl].copy()),
            "rcs_in": torch.from_numpy((rcs[sl] * mask_r).copy()),
            "hrrp_in": torch.from_numpy(hrrp[sl].copy()),
            "rcs_gt": torch.from_numpy(tr.gt_rcs[sl].copy()),
            "mask_rcs": torch.from_numpy(mask_r),
            "mask_hrrp": torch.from_numpy(tr.mask_hrrp[sl].astype(np.float32)),
            "valid": torch.ones(self.W),
            "skip_sac": True,
            "label": torch.tensor(y, dtype=torch.long),
            "traj_id": torch.tensor(tr.traj_id, dtype=torch.long),
        }
