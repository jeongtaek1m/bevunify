"""Per-model input+GT verification viz. For EACH model's experiment (its own data /
gt-toggle / split_intrin_extrin config), render N samples: the 6 camera images that
model receives + the GT signals it is actually trained against (vehicle seg, and
center / offset where that model's loss consumes them).

    cd /home/jeongtae/bevseg/bevunify
    $PY viz_samples.py                 # all models -> viz/<model>/sample_XX.png
    $PY viz_samples.py cvt lss         # subset

Note: the 6-cam images are identical across models (one unified dataloader); what
differs per model is the GT signal set (the gt/ toggle). GT BEV is front-up / left-left.
LSS/simple_bev/PointBeV outputs are flipped/transposed in their wrappers to land on
THIS GT frame (see wrappers); this panel shows the common GT they target.
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bevunify  # noqa: E402
from hydra import initialize_config_dir, compose  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from bevunify.common import setup_config, setup_data_module  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
OUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "viz")
CAM_NAMES = ["FRONT_LEFT", "FRONT", "FRONT_RIGHT", "BACK_LEFT", "BACK", "BACK_RIGHT"]
ALL_MODELS = ["gaussianlss", "cvt", "lara", "lss", "pointbev", "simplebev"]
N = 10


def to_img(t):
    return t.permute(1, 2, 0).clamp(0, 1).numpy()


def colorize_vis(mask, vis):
    """RGB GT: red = vis==1 (ignored by the vis>=2 loss), green = vis>=2 (supervised)."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    veh = mask > 0.5
    rgb[veh & (vis == 1)] = (1.0, 0.25, 0.25)                 # vis1 -> red
    rgb[veh & (vis >= 2) & (vis != 255)] = (0.3, 1.0, 0.45)   # vis>=2 -> green
    return rgb


def gt_panels(s, key):
    """list of (title, image, cmap, vmax) for the GT signals present. cmap=None -> RGB."""
    visk = f"{key}_visibility"
    if visk in s:
        first = ("GT  red=vis1(ignored)  green=vis>=2   front^ left<",
                 colorize_vis(s[key][0].numpy(), s[visk].numpy()), None, 1.0)
    else:
        first = (f"GT {key}  front^ left<", s[key][0].numpy(), "magma", 1.0)
    panels = [first]
    if f"{key}_center" in s:
        panels.append(("GT center", s[f"{key}_center"][0].numpy(), "inferno", 1.0))
    if f"{key}_offset" in s:
        off = s[f"{key}_offset"].numpy()                     # (2,200,200)
        mag = np.linalg.norm(off, axis=0)
        panels.append(("GT offset |·|", mag, "viridis", float(mag.max() or 1)))
    return panels


def render(s, idx, exp, out_dir, key):
    imgs = s["image"]
    panels = gt_panels(s, key)
    ng = len(panels)
    fig = plt.figure(figsize=(9 + 3.2 * ng, 5))
    gs = fig.add_gridspec(2, 3 + ng, width_ratios=[1, 1, 1] + [1.4] * ng,
                          wspace=0.06, hspace=0.12)
    grid = [[0, 1, 2], [3, 4, 5]]
    for r in range(2):
        for c in range(3):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(to_img(imgs[grid[r][c]]))
            ax.set_title(CAM_NAMES[grid[r][c]], fontsize=8); ax.axis("off")
    for j, (title, im, cmap, vmax) in enumerate(panels):
        ax = fig.add_subplot(gs[:, 3 + j])
        if cmap is None:
            ax.imshow(im, origin="upper")                 # RGB (visibility-colored GT)
        else:
            ax.imshow(im, cmap=cmap, vmin=0, vmax=vmax, origin="upper")
        ax.scatter([100], [100], c="cyan", s=16, marker="^")
        ax.set_title(title, fontsize=9); ax.axis("off")
    fig.suptitle(f"[{exp}] sample {idx:02d}  token={s['token'][:10]}…  | 6 cam  |  GT signals",
                 fontsize=10)
    out = os.path.join(out_dir, f"sample_{idx:02d}.png")
    fig.savefig(out, dpi=105, bbox_inches="tight"); plt.close(fig)
    return out


def run_model(exp, extra=()):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=CONFIG_DIR):
        cfg = compose(config_name="config",
                      overrides=[f"+experiment={exp}", "experiment.save_dir=/tmp/bevunify_viz/", *extra])
        setup_config(cfg)
        dm = setup_data_module(cfg)
        scenes = dm.get_split("val", loader=False)
    out_dir = os.path.join(OUT_ROOT, exp); os.makedirs(out_dir, exist_ok=True)
    picks = [(scenes[int(i)], 0) for i in np.linspace(0, len(scenes) - 1, N).astype(int)]
    sig = None
    for i, (ds, idx) in enumerate(picks):
        s = ds[idx]
        if sig is None:
            sig = [k for k in (cfg.key, f"{cfg.key}_center", f"{cfg.key}_offset") if k in s]
        render(s, i, exp, out_dir, cfg.key)
    print(f"[{exp}] {N} panels -> {out_dir}   GT signals: {sig}")


def main():
    # usage: viz_samples.py [model ...] [hydra.overrides=...]
    #   e.g. viz_samples.py cvt data.dataset_dir=/abs/nuscenes data.labels_dir=/abs/labels
    args = sys.argv[1:]
    models = [a for a in args if a in ALL_MODELS] or ALL_MODELS
    extra = [a for a in args if "=" in a]
    for m in models:
        run_model(m, extra)


if __name__ == "__main__":
    main()
