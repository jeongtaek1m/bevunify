"""Unified evaluation entry — one file, three protocols, shared by all 6 models.

Same GT + same IoU metric + same paper naming → fair cross-model comparison.

Protocols (selected via ``+eval.protocol=<name>`` or default ``normal``):

  normal   — single-pass validate over the val DB (Lightning trainer.validate)
  vr       — viewpoint-robustness 631-config grid (Normal + ER/VR/CR × 3 axes
             × 10 signed mags × {6 per-cam, 1 all-cam}), mVRS aggregation.
             Carla only.
  cts      — cross-platform transfer (NORMAL/EXT/IMG/CAL), CTS = IoU_c/oracle.
             Carla only.

Usage::

  # 1. plain validate (any dataset, any model)
  python -m bevunify.eval +experiment=cvt ckpt=/path/last.ckpt

  # 2. viewpoint-robustness 631-config eval (carla)
  python -m bevunify.eval +experiment=cvt data=carla \\
      ckpt=/path/sedan_last.ckpt +eval.protocol=vr \\
      +eval.out_dir=./eval_results/vr_sedan

  # 3. cross-platform transfer (carla sedan→suv)
  python -m bevunify.eval +experiment=cvt data=carla \\
      ckpt=/path/sedan_last.ckpt +eval.protocol=cts \\
      +eval.target_platform=suv \\
      data.val_version=v1.0-carla_suv_eval \\
      data.val_labels_dir=/.../labels/gaussianlss/suv_eval \\
      +eval.sedan_labels_dir=/.../labels/gaussianlss/sedan_eval \\
      +eval.oracle_iou=0.4123
"""
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from bevunify.common import setup_config, setup_experiment, load_backbone

log = logging.getLogger(__name__)

CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config")
CONFIG_NAME = "config.yaml"


# ── Protocol constants ────────────────────────────────────────────────────────

# Normal / ER (extrinsic-only) / VR (image-only, PRIMARY) / CR (both)
CONDITION_FLAGS = {
    "Normal": dict(use_variant=False, image_swap=False, extrinsic_swap=False),
    "ER":     dict(use_variant=True,  image_swap=False, extrinsic_swap=True),
    "VR":     dict(use_variant=True,  image_swap=True,  extrinsic_swap=False),
    "CR":     dict(use_variant=True,  image_swap=True,  extrinsic_swap=True),
}
# Paper naming for mVRS tables
PAPER = {"Normal": "Normal", "ER": "EXT", "VR": "IMG", "CR": "CAL"}
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]

# (img_to_sedan, ext_to_sedan)
CTS_CONDITIONS = {
    "NORMAL": (True,  True),
    "EXT":    (True,  False),
    "IMG":    (False, True),
    "CAL":    (False, False),
}


# ── Shared inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model, loader, threshold=0.5):
    """Returns (iou_vis_gt2, iou_vis_all). Vehicle channel only, sigmoid@thr."""
    eps = 1e-9
    tp_v = fp_v = fn_v = 0.0
    tp_a = fp_a = fn_a = 0.0
    for batch in loader:
        batch_cuda = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred = model(batch_cuda)
        logit = pred["vehicle"] if "vehicle" in pred else pred["bev"]
        if logit.ndim == 4:
            logit = logit[:, 0]                                # (B, H, W)
        prob = torch.sigmoid(logit).cpu()
        gt = batch["vehicle"][:, 0].cpu()                       # (B, H, W) {0,1}
        vis = batch["vehicle_visibility"].cpu()                 # (B, H, W) uint8

        pred_bin = (prob > threshold).float()
        gt_bin = (gt > 0.5).float()
        mask_v = (vis >= 2).float()

        tp_v += (pred_bin * gt_bin * mask_v).sum().item()
        fp_v += (pred_bin * (1 - gt_bin) * mask_v).sum().item()
        fn_v += ((1 - pred_bin) * gt_bin * mask_v).sum().item()
        tp_a += (pred_bin * gt_bin).sum().item()
        fp_a += (pred_bin * (1 - gt_bin)).sum().item()
        fn_a += ((1 - pred_bin) * gt_bin).sum().item()

    iou_v = tp_v / (tp_v + fp_v + fn_v + eps)
    iou_a = tp_a / (tp_a + fp_a + fn_a + eps)
    return iou_v, iou_a


def _ds_iter(loader):
    """Yield every CarlaGeneratedDataset / NuScenesGeneratedDataset inside the loader's dataset."""
    ds = loader.dataset
    return ds.datasets if hasattr(ds, "datasets") else [ds]


# ── VR (viewpoint-robustness) ─────────────────────────────────────────────────

def _variant_key(axis, mag):
    parts = {"yaw": 0, "pitch": 0, "roll": 0}
    parts[axis] = mag
    return f"yaw{parts['yaw']}pitch{parts['pitch']}roll{parts['roll']}"


def _build_vr_grid(axes, magnitudes, conditions, protocols):
    out = [dict(name="Normal", condition="Normal", variant=None,
                axis=None, mag=None, protocol="clean", target_camera=None)]
    for cond in conditions:
        for ax in axes:
            for mg in magnitudes:
                v = _variant_key(ax, mg)
                if "per_cam" in protocols:
                    for cam in CAMS:
                        out.append(dict(
                            name=f"{cond}_perCam_{cam}_{ax}{mg:+d}",
                            condition=cond, variant=v, axis=ax, mag=mg,
                            protocol="per_cam", target_camera=cam,
                        ))
                if "all_cam" in protocols:
                    out.append(dict(
                        name=f"{cond}_allCam_{ax}{mg:+d}",
                        condition=cond, variant=v, axis=ax, mag=mg,
                        protocol="all_cam", target_camera="ALL",
                    ))
    return out


def _mutate_vr(loader, cfg_item, viewpoint_metadata_path):
    flags = CONDITION_FLAGS[cfg_item["condition"]]
    variant = cfg_item["variant"] if flags["use_variant"] else None
    tgt = cfg_item.get("target_camera")
    target_set = None if tgt in (None, "ALL") else {tgt}
    for d in _ds_iter(loader):
        t = d.transform
        t.eval_viewpoint_variant = variant
        t.eval_image_swap = bool(flags["image_swap"])
        t.eval_extrinsic_swap = bool(flags["extrinsic_swap"])
        t.eval_target_cameras = target_set
        if variant is not None and getattr(t, "viewpoint_metadata", None) is None:
            t.viewpoint_metadata = json.loads(Path(viewpoint_metadata_path).read_text())
            t.vr_root = Path(t.viewpoint_metadata["vr_root"])


def _aggregate_mvrs(results):
    normal = next(r for r in results if r["condition"] == "Normal")
    M_normal = max(normal["iou_vis_gt2"], 1e-9)

    def rrs(M): return M / M_normal

    def sel(condition, protocol):
        for r in results:
            if r["condition"] == "Normal": continue
            if r["condition"] != condition: continue
            if r["protocol"] != protocol: continue
            yield r

    def avg(xs): return float(np.mean(xs)) if xs else float("nan")

    tables = dict(normal_iou=normal["iou_vis_gt2"])
    for cond in ("ER", "VR", "CR"):
        per_cam = avg([rrs(r["iou_vis_gt2"]) for r in sel(cond, "per_cam")])
        all_cam = avg([rrs(r["iou_vis_gt2"]) for r in sel(cond, "all_cam")])
        mvrs = 0.5 * (per_cam + all_cam) if not (np.isnan(per_cam) or np.isnan(all_cam)) else float("nan")
        tables[f"mRRS_{PAPER[cond]}_perCam"] = per_cam
        tables[f"RRSALL_{PAPER[cond]}_allCam"] = all_cam
        tables[f"mVRS_{PAPER[cond]}"] = mvrs
    return tables


# ── CTS (cross-platform transfer) ─────────────────────────────────────────────

def _mutate_cts(loader, cond, target_platform, ext_override):
    img_to_sedan, ext_to_sedan = CTS_CONDITIONS[cond]
    for d in _ds_iter(loader):
        t = d.transform
        t.cts_platform = target_platform
        t.cts_img_to_sedan = bool(img_to_sedan)
        t.cts_ext_override = ext_override if ext_to_sedan else None
        # VR perturb hooks off (CTS uses target DB without per-cam viewpoint perturb)
        t.eval_viewpoint_variant = None
        t.eval_image_swap = False
        t.eval_extrinsic_swap = False
        t.eval_target_cameras = None


def _build_sedan_ext_map(sedan_labels_dir):
    """Read one sedan_eval JSON → flat {cam_channel: 4x4 sedan extrinsic}.
    Carla per-platform extrinsics are scene-invariant (rig rigid wrt ego)."""
    p = Path(sedan_labels_dir)
    if not p.exists():
        log.warning(f"[eval/cts] sedan labels not found: {p}")
        return None
    first = next(iter(sorted(p.glob("scene_*.json"))), None)
    if first is None:
        return None
    samples = json.loads(first.read_text())
    if not samples:
        return None
    s = samples[0]
    return {ch: np.array(e, dtype=np.float32) for ch, e in zip(s.get("cam_channels", []), s["extrinsics"])}


# ── Viz (shared by vr & cts) ──────────────────────────────────────────────────

@torch.no_grad()
def _viz_subset(model, loader, viz_dir, viz_cfgs, mutate_fn, samples_per_scene=5):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    viz_dir = Path(viz_dir); viz_dir.mkdir(parents=True, exist_ok=True)

    keep = []
    for ds in _ds_iter(loader):
        n = len(ds)
        step = max(1, n // samples_per_scene)
        keep.extend((ds, i) for i in list(range(0, n, step))[:samples_per_scene])

    for cfg_item in viz_cfgs:
        mutate_fn(cfg_item)
        for ds, idx in keep:
            sample = ds[idx]
            batch_cuda = {k: (v.unsqueeze(0).cuda() if torch.is_tensor(v) else v)
                          for k, v in sample.items()}
            pred = model(batch_cuda)
            logit = pred["vehicle"] if "vehicle" in pred else pred["bev"]
            prob = torch.sigmoid(logit[0, 0]).cpu().numpy()
            gt = sample["vehicle"][0].cpu().numpy()
            vis = sample["vehicle_visibility"].cpu().numpy()
            mask = (vis >= 2).astype(np.float32)
            imgs = sample["image"].permute(0, 2, 3, 1).cpu().numpy()
            imgs = np.clip((imgs - imgs.min()) / max(imgs.max() - imgs.min(), 1e-9), 0, 1)

            fig = plt.figure(figsize=(22, 9))
            gs = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 1.7, 1.7])
            titles = ["FL", "F", "FR", "BL", "B", "BR"]
            for i in range(6):
                ax = fig.add_subplot(gs[i // 3, i % 3])
                ax.imshow(imgs[i]); ax.axis("off"); ax.set_title(titles[i], fontsize=8)
            ax_gt = fig.add_subplot(gs[:, 3]); ax_pred = fig.add_subplot(gs[:, 4])
            ax_gt.imshow(gt * mask, cmap="gray", vmin=0, vmax=1); ax_gt.axis("off")
            ax_gt.set_title("GT (vis>=2)", fontsize=10)
            ax_pred.imshow(prob, cmap="plasma", vmin=0, vmax=1); ax_pred.axis("off")
            ax_pred.set_title("Pred", fontsize=10)
            fig.suptitle(cfg_item.get("name", ""), fontsize=11)
            fig.savefig(viz_dir / f'{sample["token"]}_{cfg_item["name"]}.png', dpi=70)
            plt.close(fig)
    log.info(f"[eval] viz saved → {viz_dir}")


# ── Protocol drivers ──────────────────────────────────────────────────────────

def _run_normal(cfg, model_module, data_module):
    logger = pl.loggers.CSVLogger(save_dir=cfg.experiment.save_dir, name=cfg.experiment.project)
    trainer = pl.Trainer(logger=logger, **cfg.trainer)
    trainer.validate(model_module, datamodule=data_module)


def _run_vr(cfg, eval_cfg, model_module, data_module, out_dir):
    model = model_module.backbone.eval().cuda()
    data_module.setup("validate")
    val_loader = data_module.val_dataloader()

    grid = _build_vr_grid(eval_cfg.axes, eval_cfg.magnitudes,
                          eval_cfg.conditions, eval_cfg.protocols)
    log.info(f"[eval/vr] total configs: {len(grid)}")

    results = []
    for cfg_item in tqdm(grid, desc="VR configs"):
        _mutate_vr(val_loader, cfg_item, eval_cfg.viewpoint_metadata_path)
        iou_v, iou_a = run_inference(model, val_loader, threshold=eval_cfg.threshold)
        rec = dict(cfg_item); rec["iou_vis_gt2"] = iou_v; rec["iou_vis_all"] = iou_a
        results.append(rec)
        log.info(f"  [{cfg_item['name']}] vis>=2={iou_v:.4f} all={iou_a:.4f}")

    tables = _aggregate_mvrs(results)
    payload = dict(eval_cfg=OmegaConf.to_container(eval_cfg, resolve=True),
                   results=results, mVRS=tables)
    (out_dir / "eval_vr.json").write_text(json.dumps(payload, indent=2))
    log.info("[eval/vr] === mVRS summary ===")
    for k, v in tables.items():
        log.info(f"  {k}: {v:.4f}")

    if eval_cfg.viz_enable:
        viz_cfgs = [dict(name="Normal", condition="Normal", variant=None, axis=None, mag=None,
                         protocol="clean", target_camera=None)]
        for ax in eval_cfg.viz_axes:
            for mg in eval_cfg.viz_mags:
                viz_cfgs.append(dict(
                    name=f"VR_perCam_F_{ax}{mg:+d}",
                    condition="VR", variant=_variant_key(ax, mg), axis=ax, mag=mg,
                    protocol="per_cam", target_camera=eval_cfg.viz_target_camera,
                ))
        try:
            _viz_subset(model, val_loader, out_dir / "viz", viz_cfgs,
                        mutate_fn=lambda ci: _mutate_vr(val_loader, ci, eval_cfg.viewpoint_metadata_path),
                        samples_per_scene=eval_cfg.viz_samples_per_scene)
        except Exception as e:
            log.warning(f"[eval/vr] viz failed: {e}")


def _run_cts(cfg, eval_cfg, model_module, data_module, out_dir):
    ext_override = _build_sedan_ext_map(eval_cfg.sedan_labels_dir) if eval_cfg.get("sedan_labels_dir") else None
    if ext_override is None:
        log.warning("[eval/cts] eval.sedan_labels_dir empty — NORMAL/IMG fall back to no extrinsic swap (sedan extrinsic unavailable)")

    model = model_module.backbone.eval().cuda()
    data_module.setup("validate")
    val_loader = data_module.val_dataloader()

    results = {}
    for cond in eval_cfg.conditions:
        _mutate_cts(val_loader, cond, eval_cfg.target_platform, ext_override)
        iou_v, iou_a = run_inference(model, val_loader, threshold=eval_cfg.threshold)
        results[cond] = dict(iou_vis_gt2=iou_v, iou_vis_all=iou_a)
        log.info(f"  [{cond}] vis>=2={iou_v:.4f} all={iou_a:.4f}")

    summary = dict(target_platform=eval_cfg.target_platform)
    if eval_cfg.get("oracle_iou") is not None:
        oracle = float(eval_cfg.oracle_iou)
        summary["oracle_iou"] = oracle
        for cond, r in results.items():
            r["cts_vis_gt2"] = r["iou_vis_gt2"] / max(oracle, 1e-9)
        summary["CTS_IMG_primary"] = results.get("IMG", {}).get("cts_vis_gt2")
    summary["conditions"] = results

    (out_dir / "eval_cts.json").write_text(json.dumps(summary, indent=2))
    log.info("[eval/cts] === summary ===")
    for cond, r in results.items():
        marker = "  ← primary" if cond == "IMG" else ""
        ratio = f"  CTS={r.get('cts_vis_gt2', float('nan')):.4f}" if "cts_vis_gt2" in r else ""
        log.info(f"  {cond:>6}: vis>=2={r['iou_vis_gt2']:.4f} all={r['iou_vis_all']:.4f}{ratio}{marker}")

    if eval_cfg.viz_enable:
        viz_cfgs = [dict(name=f"CTS_{c}", condition=c) for c in eval_cfg.conditions]
        try:
            _viz_subset(model, val_loader, out_dir / "viz", viz_cfgs,
                        mutate_fn=lambda ci: _mutate_cts(val_loader, ci["condition"],
                                                         eval_cfg.target_platform, ext_override),
                        samples_per_scene=eval_cfg.viz_samples_per_scene)
        except Exception as e:
            log.warning(f"[eval/cts] viz failed: {e}")


# ── Defaults ──────────────────────────────────────────────────────────────────

def _default_eval_cfg(protocol, cfg):
    """Defaults per protocol (CLI can override via +eval.<key>=val)."""
    if protocol == "normal":
        return OmegaConf.create({})
    if protocol == "vr":
        return OmegaConf.create(dict(
            out_dir="${hydra:runtime.cwd}/eval_results/vr_${experiment.uuid}",
            axes=["pitch", "yaw", "roll"],
            magnitudes=[-20, -16, -12, -8, -4, 4, 8, 12, 16, 20],
            conditions=["ER", "VR", "CR"],
            protocols=["per_cam", "all_cam"],
            threshold=0.5,
            viz_enable=True,
            viz_samples_per_scene=5,
            viz_axes=["pitch", "yaw", "roll"],
            viz_mags=[-20, -8, 4, 20],
            viz_target_camera="CAM_FRONT",
            viewpoint_metadata_path=cfg.data.get("viewpoint_metadata_path", None),
        ))
    if protocol == "cts":
        return OmegaConf.create(dict(
            out_dir="${hydra:runtime.cwd}/eval_results/cts_${experiment.uuid}",
            target_platform="suv",                 # 'suv' | 'bus'
            conditions=["NORMAL", "EXT", "IMG", "CAL"],
            threshold=0.5,
            oracle_iou=None,
            sedan_labels_dir=None,
            viz_enable=False,
            viz_samples_per_scene=5,
        ))
    raise ValueError(f"unknown eval.protocol={protocol!r}")


# ── Entry ─────────────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    torch.set_float32_matmul_precision("high")
    setup_config(cfg)

    protocol = cfg.get("eval", {}).get("protocol", "normal") if cfg.get("eval") else "normal"
    eval_cfg = _default_eval_cfg(protocol, cfg)
    if cfg.get("eval"):
        eval_cfg = OmegaConf.merge(eval_cfg, cfg.eval)

    model_module, data_module, _ = setup_experiment(cfg)
    ckpt_path = cfg.get("ckpt", None)
    if ckpt_path is not None:
        model_module.backbone = load_backbone(ckpt_path, backbone=model_module.backbone)

    if protocol == "normal":
        _run_normal(cfg, model_module, data_module)
        return

    out_dir = Path(hydra.utils.to_absolute_path(eval_cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"[eval/{protocol}] output → {out_dir}")

    if protocol == "vr":
        _run_vr(cfg, eval_cfg, model_module, data_module, out_dir)
    elif protocol == "cts":
        _run_cts(cfg, eval_cfg, model_module, data_module, out_dir)


if __name__ == "__main__":
    main()
