# SP-ARF — Structure-Preserving Asynchronous Radar Fusion

Reference implementation of

> **Structure-Preserving Multimodal Learning for Asynchronous RCS–HRRP Radar
> Target Recognition**, Pattern Recognition.

RCS-related amplitude sequences and High-Resolution Range Profiles (HRRPs) are
acquired at **non-coincident timestamps** by a single-station multifunction
radar, so paired observations do not exist. SP-ARF completes each modality in a
structure-preserving way and then fuses the completed pair:

| Module | Role |
|---|---|
| **TDC** — Temporal-Dynamics Completion | trend-conditioned latent Neural ODE (RK2) + observation-driven gated correction for the irregular RCS-related amplitude sequence |
| **SAC** — Scattering-Aligned Completion | parameter-free HRRP completion by range-bin shift estimation and structural-similarity-gated interpolation / nearest-frame inheritance |
| **EAF** — Energy-Guided Adaptive Fusion | bidirectional cross-modal attention biased by HRRP scattering energy, followed by adaptive modality gating |

---

## Layout

```
sp_arf/
  config.py         hyper-parameters
  data.py           loading, trajectory-level split, completion, windowing
  engine.py         training / evaluation
  metrics.py        accuracy, macro P-R-F1, kappa
  models/
    tdc.py          Temporal-Dynamics Completion  (Section 3.3)
    sac.py          Scattering-Aligned Completion (Section 3.4)
    eaf.py          Energy-Guided Adaptive Fusion (Section 3.5)
    sp_arf.py       assembled network + ablation levels 1-7
    loss.py         training objective           (Section 3.6)
scripts/
  train.py          training
  evaluate.py       evaluate a checkpoint
  verify_release.py check that the released weights reproduce 89.11 %
checkpoints/
  SP-ARF.pth        released weights, 0.6004 M parameters
```

## Install

```bash
pip install -r requirements.txt        # torch>=2.0, numpy, scikit-learn
```

Developed with Python 3.8 and PyTorch 2.0.1 on a single RTX 3060; training also
runs on CPU, roughly 20x slower.

## Dataset

The benchmark is distributed separately and is **not** included. Place the 90
`.npy` files as follows:

```
data/<class>_traj_<1..5>_async.npy    # asynchronous observations (network input)
data/<class>_traj_<1..5>_sync.npy     # synchronous reference (evaluation only)
```

* 9 classes — `EA-18G, EP-3E, F2, F15, F16, F18, F22, F35, IDF` (**this order
  defines the label indices** and must not be re-sorted)
* 5 trajectories per class, 3,000 simulation steps each, 256 range cells,
  X-band co-polarised HH channel
* asynchronous files: `rcs_time`, `rcs_data`, `hrrp_time`, `hrrp_data`,
  `target_type`, `total_steps`
* the synchronous files are used only as the reconstruction/evaluation reference
  and are never fed to the network

Pass `--data /path/to/data` if the directory lives elsewhere.

## Citation

```bibtex
@article{liao2024sparf,
  title   = {Structure-Preserving Multimodal Learning for Asynchronous
             {RCS--HRRP} Radar Target Recognition},
  author  = {Liao, Huaizhang and Liu, Yun and Zhang, Entao and Xia, Jingyuan
             and Liu, Yongxiang},
  journal = {Pattern Recognition}
}
```

## License

MIT for the code — see `LICENSE`. The released weights and the dataset follow the
terms of the corresponding publication and data-release agreement.
