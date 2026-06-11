# EAFormer (fork of Cross View Transformers)

Reimplementation of **EAFormer** — *Epipolar Attention Field Transformers for Bird's Eye
View Semantic Segmentation*, Witte, Behley, Stachniss, Raaijmakers, **WACV 2025**
([arXiv:2412.01595](https://arxiv.org/abs/2412.01595)). There is **no official code
release**; this is a from-paper reimplementation built as a fork of
[bradyz/cross_view_transformers](https://github.com/bradyz/cross_view_transformers)
(Brady Zhou & Philipp Krähenbühl, CVPR 2022, MIT license — see LICENSE).

The python package is renamed `cross_view_transformer` → `eaformer` so both models can
be imported in one process (the original CVT stays vendored at
`third_party/cross_view_transformers`).

## What changed vs CVT

- `eaformer/model/encoder.py` — **Epipolar Attention Fields**: per (BEV-cell, camera)
  epipolar line (two-point vertical-line projection, shared geometry hoisted to
  `Encoder.forward`); Gaussian log-W over perpendicular pixel distance with
  σ_px = 1/(λ·λ_qi), λ_qi=(d+d0)/range (near = wide footprint), learnable λ. Applied
  as the paper's Eq. 2 multiplicative gate `softmax(W ⊙ QKᵀ/√d)`; behind-camera keys
  hard-masked additively. Keys at true feature-cell centers; no positional encoding by
  default (`use_pe`/`use_eaf`/`eaf_*` switchable in `config/model/eaformer.yaml`).
- `eaformer/model/decoder.py` — decoder with **three ASPP blocks** (paper supp. Sec. 1)
  + CVT-style long-range skip + full-resolution refine conv.
- `config/model/eaformer.yaml`, `config/experiment/eaformer_nuscenes_vehicle.yaml` —
  fork-side hydra configs. `config/model/cvt.yaml` pins `use_eaf: False, use_pe: True`
  (plain-CVT path inside the fork).

Integration with the unified pipeline lives outside this fork:
`bevunify/wrappers/eaformer.py` + `config/{model,experiment,recipe}/eaformer.yaml`.

## Training protocols (nuScenes vehicle, this repo's GT/metric)

| protocol | command | IoU@0.5 (vis≥2) |
|---|---|---|
| paper-exact: 1 GPU, eff. batch 16, lr 4e-3, 30 ep | `... trainer.devices=1` | 0.370 |
| 2-GPU: eff. batch 32, lr 5.7e-3 (√2), 60 ep (update-matched) | `... optimizer.lr=5.7e-3 trainer.max_epochs=60` | 0.371 |

CVT baseline, same pipeline/recipe (paper-exact protocol): 0.370. Unified comparisons
use **IoU@0.5 only**; the papers' max-IoU-over-{0.4,0.45,0.5} number is produced
post-hoc by `tests/eval_threshold_sweep.py` and must not be mixed into the @0.5 table.
