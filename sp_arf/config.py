"""Hyper-parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data"


@dataclass
class DataConfig:
    num_classes: int = 9
    hrrp_dim: int = 256
    total_steps: int = 3000
    trajectories_per_class: int = 5
    train_traj_ids: Tuple[int, ...] = (1, 2, 3)
    test_traj_ids: Tuple[int, ...] = (4, 5)
    pol: str = "CoHH"
    normalise: bool = False


@dataclass
class WindowConfig:
    window: int = 40
    stride: int = 20


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 8
    d_z: int = 64
    d_c: int = 64
    d_e: int = 64
    d_v: int = 128
    delta_max: int = 15
    rho_th: float = 0.7
    dropout: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 24
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    lambda_rec: float = 0.1
    lambda_dyn: float = 0.1
    seed: int = 42
    device: str = "cuda"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
