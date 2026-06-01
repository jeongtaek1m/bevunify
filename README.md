# bevunify

One **Hydra** project to train/evaluate **6 camera-only BEV-segmentation models on the
same GaussianLSS ground truth**, switched with a single flag:

```bash
cd bevunify
PY=/path/to/envs/GaussianLSS/bin/python      # the host conda env (torch + hydra + nuscenes)

$PY -m bevunify.train +experiment=gaussianlss     # GaussianLSS (host model)
$PY -m bevunify.train +experiment=cvt             # cross_view_transformers
$PY -m bevunify.train +experiment=lara            # LaRa
$PY -m bevunify.train +experiment=lss             # lift-splat-shoot
$PY -m bevunify.train +experiment=pointbev        # PointBeV
$PY -m bevunify.train +experiment=simplebev       # simple_bev
```

**GaussianLSS is the host**: its `LoadDataTransform`, `DataModule`, `MultipleLoss`,
`IoUMetric`, `ModelModule` are reused unchanged. Every other model is wrapped as an
`nn.Module` that takes the GaussianLSS batch and returns the canonical prediction dict
(`{vehicle, vehicle_center, vehicle_offset}`) **in the GaussianLSS BEV frame**, so it
plugs straight into the host Lightning spine. One GT, one metric → fair comparison.

---

## Folder / file guide

```
bevunify/
├── README.md
├── setup.py                 # pip metadata (pip install -e .)
├── verify_setup.py          # static check: composes all 6 experiments + imports every module
├── viz_samples.py           # per-model input+GT viz: 6 cam | GT signals  ->  viz/<model>/sample_*.png
│
├── config/                  # ── Hydra config tree (the "one switch" lives here) ──
│   ├── config.yaml          #   root defaults list + experiment/loader/trainer/optimizer + key/min_visibility
│   ├── data/
│   │   ├── nuscenes.yaml     #   SHARED dataloader: dataset paths, BEV grid (200x200/±50m), image res, augment flags
│   │   └── img_params/scale_0_3.yaml   # image-aug params (only used when augment_img=True)
│   ├── gt/                   #   per-model GT-signal TOGGLE (package: data) — generate only what the loss needs
│   │   ├── seg_only.yaml             # seg (+visibility)
│   │   ├── seg_center.yaml           # + center heatmap
│   │   └── seg_center_offset.yaml    # + center + offset
│   ├── loss/                 #   shared GaussianLSS losses composed per toggle
│   │   ├── seg.yaml / seg_center.yaml / seg_center_offset.yaml
│   ├── metrics/
│   │   └── bev_metrics.yaml  #   unified metric: IoU@0.5 at vis>=2 AND vis-all (all models)
│   ├── model/                #   one file per model -> _target_ (host model or a bevunify wrapper)
│   │   ├── gaussianlss.yaml  cvt.yaml  lara.yaml  lss.yaml  pointbev.yaml  simplebev.yaml
│   ├── recipe/               #   per-model TRAINING recipe (optimizer / scheduler / epochs) — they all differ
│   │   ├── gaussianlss.yaml(OneCycle 50ep)  cvt.yaml(OneCycle 30k)  lara.yaml(Adam+StepLR 30ep)
│   │   ├── lss.yaml(Adam const)  pointbev.yaml(OneCycle)  simplebev.yaml(OneCycle 100k)
│   └── experiment/           #   THE SWITCH: +experiment=<model> wires model+gt+loss+recipe+metric+data
│       ├── gaussianlss.yaml  cvt.yaml  lara.yaml  lss.yaml  pointbev.yaml  simplebev.yaml
│
├── bevunify/                 # ── python package (the integration code) ──
│   ├── __init__.py           #   puts the GaussianLSS host repo on sys.path (GAUSSIANLSS_ROOT)
│   ├── common.py             #   setup_experiment/model_module — reuses host spine, registers the toggle dataset
│   ├── model_module.py       #   UnifiedModelModule: configure_optimizers generalized to any optimizer/scheduler
│   │                         #     (_target_ or default) so each model uses its own recipe; OneCycle total_steps auto
│   ├── transforms_toggle.py  #   ToggleLoadDataTransform: GaussianLSS GT rasterizer with center/offset/visibility TOGGLE
│   ├── data_toggle.py        #   dataset module (get_data) that uses the toggle transform  (dataset: nuscenes_toggle)
│   ├── datagen.py            #   augmented label generation: host data-gen + per-cam ego_from_cam  ->  labels_aug
│   ├── train.py              #   @hydra.main training entry (W&B/CSV logger + val-viz callback + DDP)
│   ├── eval.py               #   evaluation entry (trainer.validate)
│   ├── viz_callback.py       #   ValVizCallback: render 6 cam | GT | pred every N val steps -> W&B + PNG
│   └── wrappers/             #   per-model adapters: GaussianLSS batch -> model.forward -> canonical pred (GT frame)
│       ├── geom.py           #     extrinsic inversion (cam->ego, prefers materialized ego_from_cam) + intrinsics helpers
│       ├── repo_compose.py   #     compose a source repo's OWN hydra config in isolation (reuse its exact architecture)
│       ├── cvt.py            #     CVT: dict in/out; emits vehicle(+center)
│       ├── lara.py           #     LaRa: imgs+rots/trans+intrins; output flip(-2,-1) to GT frame
│       ├── lss.py            #     LSS:  output flip(-2,-1) to GT frame
│       ├── pointbev.py       #     PointBeV: single-frame; output already in GT frame; offset channels swapped
│       └── simplebev.py      #     simple_bev (camera-only): output flip(-2); offset flip(-2)+negate ch0
│
├── tests/
│   ├── smoke_forward.py      #   end-to-end smoke for one experiment: data -> model -> loss -> metric on 1 batch
│   └── probe_orientation.py  #   orientation-probe scaffold
│
├── viz/   (gitignored)       #   generated input/GT/pred PNGs
└── logs/  (gitignored)       #   training logs, checkpoints, val_viz PNGs
```

---

## Ground truth (single source, per-model toggle)

GT = GaussianLSS labels at `data.labels_dir` (default `labels_aug`, which also stores
per-camera `ego_from_cam`). BEV is rasterised on-the-fly at **200×200 / ±50 m / 0.5 m-px**,
ego-centric, `row = -X (front)`, `col = -Y (left)`. `ToggleLoadDataTransform` emits only
the signals a model's loss consumes:

| GT preset | seg | center | offset | visibility | models |
|---|:--:|:--:|:--:|:--:|---|
| `seg_only`          | ✓ | – | – | ✓ | lift-splat-shoot, LaRa |
| `seg_center`        | ✓ | ✓ | – | ✓ | cross_view_transformers |
| `seg_center_offset` | ✓ | ✓ | ✓ | ✓ | GaussianLSS, PointBeV, simple_bev |

Image resolution & augmentation are managed **centrally** in `config/data/nuscenes.yaml`
(`image.{h,w,top_crop}`, `augment_img`, `augment_bev`) — currently **224×480, augmentation OFF**.

---

## Per-model status

| model | wrapper | output→GT axis fix | batch | geometry verified |
|---|---|---|---|---|
| GaussianLSS | host model | native | 16 | host (native) |
| cross_view_transformers | `wrappers/cvt.py` | none | 16 | ✓ (train IoU↑) |
| lift-splat-shoot | `wrappers/lss.py` | `flip(-2).flip(-1)` | 16 | ✓ (train IoU↑) |
| LaRa | `wrappers/lara.py` | `flip(-2).flip(-1)` | **4** (OOM at 16) | in progress |
| PointBeV | `wrappers/pointbev.py` | none + offset ch-swap | 16 | in progress |
| simple_bev | `wrappers/simplebev.py` | `flip(-2)` + offset negate ch0 | 16 | in progress |

Axis fixes were derived by a coordinate-convention audit (V-matrix vs each model's output
grid) and confirmed by training (IoU rises only when aligned).

---

## Setup (prerequisites)

bevunify imports the 6 source repos **in place** — clone them beside this repo and point
`GAUSSIANLSS_ROOT` / the `repo_root` fields in `config/model/*.yaml` at them:

```
<workspace>/
  GaussianLSS/  cross_view_transformers/  LaRa/  lift-splat-shoot/  PointBeV/  simple_bev/
  bevunify/     <- this repo
```

Host env (one conda env, Python 3.8). Install PyTorch (CUDA) + the PointBeV sparse-conv
wheel first, then the rest from `requirements.txt`:
```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install spconv-cu120==2.3.6
pip install -r requirements.txt          # hydra, lightning, nuscenes-devkit, fairscale, rich, ...
```

Extra steps:
- **GaussianLSS** rasterizer (`diff-gaussian-rasterization`) built.
- **PointBeV**: add `from __future__ import annotations` to `pointbev/models/sampled.py`
  (repo uses Py3.10 `X|Y` syntax); build its CUDA op:
  `cd PointBeV/pointbev/ops/gs && CUDA_HOME=/usr/local/cuda-12.1 python setup.py build install`.
- **LaRa**: set `WEIGHTS_PATH` env (cam-encoder weights).
- **Data**: generate GaussianLSS labels (with `ego_from_cam`):
  `python -m bevunify.datagen --dataset_dir <nuscenes> --labels_dir <nuscenes>/labels_aug`.

## Usage

```bash
# verify config graph + imports (no GPU)
python verify_setup.py

# per-model input+GT visualization
python viz_samples.py [cvt lss ...]          # -> viz/<model>/

# train (W&B per-model project bevseg-<model>; experiment.logger=csv to disable W&B)
python -m bevunify.train +experiment=<model> trainer.devices=2

# evaluate
python -m bevunify.eval +experiment=<model> ckpt=<path>
```

Training logs a `6 cam | GT | pred` panel every `experiment.val_viz_interval` (100) val
steps to W&B and to `logs/val_viz/`. Validation is **not** shuffled.
