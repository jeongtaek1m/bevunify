"""Per-model empirical probe: does the VR/ER extrinsic swap reach the model's
BEV pred output (not just the data layer)?

For each of the 6 BEV models we:

  1. Build a hydra cfg (``+experiment=<m>  data=carla  data.version=v1.0-carla_sedan``).
  2. Instantiate ``model_module`` via ``bevunify.common.setup_experiment(cfg)``.
  3. Move backbone to device (CUDA if available else CPU; --device overrides).
  4. Load scene_0220 frame 0 from sedan_eval.
  5. Run Normal forward          → pred_normal['vehicle']    (B, 1, 200, 200)
  6. Mutate the dataset transform to ER (yaw=-20, target={CAM_FRONT}):
        eval_viewpoint_variant = 'yaw-20pitch0roll0'
        eval_image_swap        = False
        eval_extrinsic_swap    = True
        eval_target_cameras    = {'CAM_FRONT'}
  7. Reload the SAME sample      → pred_er
  8. Compare pred_normal vs pred_er (MSE, L1, max|Δ|, IoU on bin@0.5).
  9. Assert MSE > 1e-4  — the extrinsic swap empirically reaches the model.
 10. Save a 4-panel viz to /tmp/probe_per_model_<m>.png:
       (CAM_FRONT_normal | CAM_FRONT_normal_repeat | pred_normal | pred_er)

This complements ``probe_vr_extrinsic_swap.py`` (data-layer math) by closing the
loop end-to-end: extrinsic change in → pred change out, per model.

Notes
-----
- This is a *forward-only* probe, no checkpoint, no loss. Random-init models
  still respond to extrinsic perturbation — the check is differential, not
  absolute. (Untrained CVT/PointBeV/etc. will produce noise BEVs, but Normal
  vs ER noise will differ because the geometry input differs.)
- A model whose env (e.g. ``pointbev_b200``) is unavailable in the current
  interpreter is recorded as **SKIPPED**, not FAIL.
- CUDA OOM → falls back to CPU forward for that one model.
- Each model is wrapped in try/except: one model crashing does NOT abort
  the rest.

Run (after GPU frees):
  /home/hanyan_arch/miniconda3/envs/GaussianLSS/bin/python \\
      /home/hanyan_arch/git/BEV_seg/bevunify/tests/probe_per_model_extrinsic_forward.py
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

try:
    _THIS = Path(__file__).resolve()
except NameError:
    # `exec(open(...).read())` strips __file__; the user's import-sanity
    # check runs the script that way, so fall back to the known repo path.
    _THIS = Path("/home/hanyan_arch/git/BEV_seg/bevunify/tests/probe_per_model_extrinsic_forward.py")
REPO_ROOT = _THIS.resolve().parents[1]
HOST_ROOT = REPO_ROOT / "third_party" / "GaussianLSS"
for p in (str(HOST_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402

# `import bevunify` bootstraps the GaussianLSS host path; do this BEFORE hydra
# tries to resolve `bevunify.*` instantiate targets.
import bevunify                                                 # noqa: E402,F401

# ── Constants ────────────────────────────────────────────────────────────────
MODELS = ["cvt", "gaussianlss", "lara", "lss", "pointbev", "simplebev"]
SCENE = "scene_0220"
FRAME_IDX = 0
ER_VARIANT = "yaw-20pitch0roll0"
ER_TARGET_CAM = "CAM_FRONT"
CAM_FRONT_IDX = 1   # ['CAM_FRONT_LEFT', 'CAM_FRONT', ...]

CONFIG_DIR = str(REPO_ROOT / "config")
OUT_DIR = Path("/tmp")
MSE_FAIL_THRESH = 1e-4


# ── Helpers ──────────────────────────────────────────────────────────────────

@contextmanager
def _cd(path: Path):
    """Hydra resolves ${hydra:runtime.cwd} against the active cwd; pin it to /tmp
    so save_dir etc. don't pollute the repo."""
    old = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old)


def _compose_cfg(model_name: str):
    """Build the cfg for one model. Hydra Compose API (no @hydra.main)."""
    from hydra import initialize_config_dir, compose
    from bevunify.common import setup_config

    overrides = [
        f"+experiment={model_name}",
        "data=carla",
        "data.version=v1.0-carla_sedan",
        # Single sample, no real loader: keep workers minimal but the DataModule
        # still constructs successfully.
        "loader.val_batch_size=1",
        "loader.num_workers=0",
        "experiment.save_dir=/tmp/bevunify_probe/",
        "experiment.logger=csv",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=overrides)
        setup_config(cfg)   # resolve ${...} while hydra context is live
    return cfg


def _get_val_datasets(cfg):
    """Return the list of CarlaGeneratedDataset for the val split, WITHOUT
    wrapping in a DataLoader (we mutate the transform between two forwards on
    one sample — bypassing the loader avoids worker-cache staleness)."""
    from bevunify.data_toggle import get_data_carla

    data_cfg = dict(cfg.data)
    # `dataset` is the host module key (carla_toggle); get_data_carla doesn't
    # consume it. Pop to avoid TypeError.
    data_cfg.pop("dataset", None)
    return get_data_carla(split="val", **data_cfg)


def _find_dataset(datasets, scene_name: str):
    for ds in datasets:
        if ds.scene_name == scene_name:
            return ds
    raise RuntimeError(
        f"scene {scene_name!r} not in val datasets "
        f"({[d.scene_name for d in datasets[:5]]} ... {len(datasets)} total)"
    )


def _mutate_to_er(transform, cfg=None):
    """Match bevunify.eval._mutate_vr for ER@CAM_FRONT yaw=-20."""
    import json
    transform.eval_viewpoint_variant = ER_VARIANT
    transform.eval_image_swap = False
    transform.eval_extrinsic_swap = True
    transform.eval_target_cameras = {ER_TARGET_CAM}
    # Lazy-load metadata if the transform hasn't seen a non-None variant yet.
    if getattr(transform, "viewpoint_metadata", None) is None:
        meta_path = getattr(transform, "viewpoint_metadata_path", None)
        # Fallback to cfg.data.viewpoint_metadata_path (transforms built before the
        # viewpoint_metadata_path attribute was added won't expose it).
        if (meta_path is None) and (cfg is not None):
            meta_path = cfg.data.get("viewpoint_metadata_path", None)
        if meta_path is not None and Path(meta_path).exists():
            transform.viewpoint_metadata = json.loads(Path(meta_path).read_text())
            transform.vr_root = Path(transform.viewpoint_metadata["vr_root"])
        else:
            raise RuntimeError(
                f"viewpoint_metadata_path not resolvable (transform attr=None, "
                f"cfg.data.viewpoint_metadata_path={meta_path!r}) — extrinsic swap "
                f"will silently no-op in get_cameras. Fix the data config or pass "
                f"viewpoint_metadata_path through the transform ctor."
            )


def _reset_to_normal(transform):
    transform.eval_viewpoint_variant = None
    transform.eval_image_swap = False
    transform.eval_extrinsic_swap = False
    transform.eval_target_cameras = None


def _sample_to_batch(sample, device):
    return {
        k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v)
        for k, v in sample.items()
    }


def _pred_logit(pred):
    """Extract vehicle logit (B,1,H,W) → (H,W) on CPU for comparison."""
    if "vehicle" in pred:
        logit = pred["vehicle"]
    elif "bev" in pred:
        logit = pred["bev"]
    else:
        raise KeyError(f"no vehicle/bev key in pred: {list(pred.keys())}")
    # canonical shape (B,C,H,W); take batch 0 channel 0
    while logit.ndim > 2:
        logit = logit[0]
    return logit.detach().float().cpu()


def _pred_iou(a_bin: torch.Tensor, b_bin: torch.Tensor) -> float:
    inter = (a_bin * b_bin).sum().item()
    union = (a_bin + b_bin - a_bin * b_bin).sum().item()
    return inter / (union + 1e-9)


def _viz(model_name, cam_front_img, pred_normal, pred_er, out_path):
    """4-panel: (cam_front_normal | cam_front_normal_repeat | pred_normal | pred_er).
    The 'repeat' is intentional — it's a sanity panel that the same input image
    was used both times (no image swap, only extrinsic swap)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  [warn] matplotlib unavailable, skipping viz: {e}")
        return

    img = cam_front_img.permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / max(img.max() - img.min(), 1e-9)
    img = np.clip(img, 0, 1)

    prob_n = torch.sigmoid(pred_normal).numpy()
    prob_e = torch.sigmoid(pred_er).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img); axes[0].set_title("CAM_FRONT (Normal)", fontsize=9); axes[0].axis("off")
    axes[1].imshow(img); axes[1].set_title("CAM_FRONT (ER — image same)", fontsize=9); axes[1].axis("off")
    axes[2].imshow(prob_n, cmap="plasma", vmin=0, vmax=1)
    axes[2].set_title("pred_normal sigmoid", fontsize=9); axes[2].axis("off")
    axes[3].imshow(prob_e, cmap="plasma", vmin=0, vmax=1)
    axes[3].set_title("pred_er sigmoid", fontsize=9); axes[3].axis("off")
    fig.suptitle(f"{model_name}: ER {ER_VARIANT} target={ER_TARGET_CAM} | {SCENE} frame {FRAME_IDX}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Per-model probe ──────────────────────────────────────────────────────────

def _probe_one(model_name: str, device_pref: str):
    """Returns dict with keys: status, mse, iou, l1, max_abs, png, note."""
    result = dict(model=model_name, status="UNKNOWN", mse=None, iou=None,
                  l1=None, max_abs=None, png=None, note="")

    # ── 1. compose cfg + instantiate model_module ───────────────────────────
    try:
        with _cd(Path("/tmp")):
            cfg = _compose_cfg(model_name)
            from bevunify.common import setup_experiment
            model_module, _, _ = setup_experiment(cfg)
    except (ImportError, ModuleNotFoundError) as e:
        result["status"] = "SKIPPED"
        result["note"] = f"missing env/import: {e.__class__.__name__}: {e}"
        return result
    except Exception as e:
        result["status"] = "ERROR_INSTANTIATE"
        result["note"] = f"{e.__class__.__name__}: {e}"
        return result

    # ── 2. pick device (OOM → CPU) ──────────────────────────────────────────
    def _try_device(dev):
        try:
            return model_module.backbone.to(dev).eval(), dev
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:  # type: ignore[attr-defined]
            print(f"  [warn] backbone.to({dev}) failed: {e}; falling back to cpu")
            torch.cuda.empty_cache()
            return model_module.backbone.to("cpu").eval(), "cpu"

    if device_pref == "cpu" or not torch.cuda.is_available():
        backbone, device = model_module.backbone.to("cpu").eval(), "cpu"
    else:
        backbone, device = _try_device(device_pref)

    # ── 3. build dataset list and find scene_0220 ───────────────────────────
    try:
        with _cd(Path("/tmp")):
            datasets = _get_val_datasets(cfg)
        ds = _find_dataset(datasets, SCENE)
    except Exception as e:
        result["status"] = "ERROR_DATA"
        result["note"] = f"{e.__class__.__name__}: {e}"
        return result

    transform = ds.transform

    # ── 4. Normal forward ───────────────────────────────────────────────────
    try:
        _reset_to_normal(transform)
        sample_n = ds[FRAME_IDX]
        batch_n = _sample_to_batch(sample_n, device)
        with torch.no_grad():
            pred_n_full = backbone(batch_n)
        pred_n = _pred_logit(pred_n_full)
        cam_front_img = sample_n["image"][CAM_FRONT_IDX].detach().cpu()
    except torch.cuda.OutOfMemoryError:
        print("  [warn] CUDA OOM on Normal forward; retrying on CPU")
        torch.cuda.empty_cache(); gc.collect()
        backbone = backbone.to("cpu"); device = "cpu"
        sample_n = ds[FRAME_IDX]
        batch_n = _sample_to_batch(sample_n, device)
        with torch.no_grad():
            pred_n_full = backbone(batch_n)
        pred_n = _pred_logit(pred_n_full)
        cam_front_img = sample_n["image"][CAM_FRONT_IDX].detach().cpu()
    except Exception as e:
        result["status"] = "ERROR_FORWARD_NORMAL"
        result["note"] = f"{e.__class__.__name__}: {e}"
        return result

    # ── 5. ER forward ───────────────────────────────────────────────────────
    try:
        _mutate_to_er(transform, cfg=cfg)
        sample_er = ds[FRAME_IDX]
        batch_er = _sample_to_batch(sample_er, device)
        with torch.no_grad():
            pred_er_full = backbone(batch_er)
        pred_er = _pred_logit(pred_er_full)
    except Exception as e:
        result["status"] = "ERROR_FORWARD_ER"
        result["note"] = f"{e.__class__.__name__}: {e}"
        return result
    finally:
        _reset_to_normal(transform)

    # ── 6. compare ──────────────────────────────────────────────────────────
    diff = (pred_n - pred_er).float()
    mse = float((diff ** 2).mean())
    l1 = float(diff.abs().mean())
    max_abs = float(diff.abs().max())
    bin_n = (torch.sigmoid(pred_n) > 0.5).float()
    bin_e = (torch.sigmoid(pred_er) > 0.5).float()
    iou = _pred_iou(bin_n, bin_e)

    result.update(mse=mse, l1=l1, max_abs=max_abs, iou=iou)

    # ── 7. viz ──────────────────────────────────────────────────────────────
    png = OUT_DIR / f"probe_per_model_{model_name}.png"
    try:
        _viz(model_name, cam_front_img, pred_n, pred_er, png)
        result["png"] = str(png)
    except Exception as e:
        result["note"] = f"viz failed: {e.__class__.__name__}: {e}"

    # ── 8. pass/fail ────────────────────────────────────────────────────────
    if mse > MSE_FAIL_THRESH:
        result["status"] = "PASS"
    else:
        result["status"] = "FAIL"
        if not result["note"]:
            result["note"] = f"MSE {mse:.3e} <= threshold {MSE_FAIL_THRESH:.0e}"

    # free CUDA memory before next model
    del backbone, model_module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ── Driver ───────────────────────────────────────────────────────────────────

def _print_summary(results):
    print("\n" + "=" * 88)
    print(" SUMMARY")
    print("=" * 88)
    header = f"{'model':<12} {'status':<22} {'MSE':>12} {'L1':>12} {'IoU(n,er)':>10}   note"
    print(header)
    print("-" * 88)
    for r in results:
        mse_s = f"{r['mse']:.3e}" if r["mse"] is not None else "      —"
        l1_s = f"{r['l1']:.3e}" if r["l1"] is not None else "      —"
        iou_s = f"{r['iou']:.4f}" if r["iou"] is not None else "    —"
        note = r["note"]
        if len(note) > 50:
            note = note[:47] + "..."
        print(f"{r['model']:<12} {r['status']:<22} {mse_s:>12} {l1_s:>12} {iou_s:>10}   {note}")
    print("=" * 88)
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_skip = sum(1 for r in results if r["status"] == "SKIPPED")
    n_fail = len(results) - n_pass - n_skip
    print(f" {n_pass} PASS / {n_skip} SKIPPED / {n_fail} FAIL/ERROR  (of {len(results)})")
    print("=" * 88)
    return n_fail


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda",
                    help="cuda|cpu|cuda:0|cuda:1  (default: cuda, falls back to cpu on OOM)")
    ap.add_argument("--models", default=",".join(MODELS),
                    help=f"comma-list subset (default: {','.join(MODELS)})")
    return ap.parse_args()


def run():
    args = _parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        print(f"[warn] unknown model(s) {unknown}; valid: {MODELS}")
        models = [m for m in models if m in MODELS]

    print(f"[probe] device pref = {args.device}")
    print(f"[probe] models      = {models}")
    print(f"[probe] scene/frame = {SCENE} / {FRAME_IDX}")
    print(f"[probe] ER          = variant={ER_VARIANT}, target={{ {ER_TARGET_CAM} }}")
    print()

    results = []
    for m in models:
        print(f"\n{'#' * 70}\n# probing: {m}\n{'#' * 70}")
        try:
            r = _probe_one(m, args.device)
        except Exception as e:
            traceback.print_exc()
            r = dict(model=m, status="ERROR_UNCAUGHT", mse=None, iou=None,
                     l1=None, max_abs=None, png=None,
                     note=f"{e.__class__.__name__}: {e}")
        results.append(r)
        mse_s = "None" if r["mse"] is None else f"{r['mse']:.3e}"
        iou_s = "None" if r["iou"] is None else f"{r['iou']:.4f}"
        print(f"  -> {r['status']}  MSE={mse_s}  IoU={iou_s}  {r['note']}")

    n_fail = _print_summary(results)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    run()
