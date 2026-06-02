"""Generate lidar2cam projection viz for every VR config.

Grid (per scene+frame):
  axes × magnitudes × conditions × {6 per_cam target, 1 all_cam} = 630
  + 1 Normal baseline
  = 631 PNGs

Each PNG is a 2x3 grid of the 6 cams showing the lidar projected over the
(possibly swapped) image with the (possibly perturbed) extrinsic. The target
camera(s) get a red border.

Run::
  python -m bevunify.carla_data_toolkit.viz_vr_lidar2cam
  python -m bevunify.carla_data_toolkit.viz_vr_lidar2cam --scene scene_0220 \
      --frame 0 --conditions Normal ER VR CR --protocols per_cam all_cam \
      --limit 0    # 0 = no limit; else cap PNG count

Outputs to: bevunify/carla_data_toolkit/viz/vr/<scene>_f<frame>/<name>.png
"""
from __future__ import annotations
import argparse
from itertools import product
from pathlib import Path

from tqdm import tqdm

from .projection import CAMS, load_frame, load_vr_metadata, apply_vr
from .plot import render_projection


CONDITION_FLAGS = {
    "Normal": dict(image_swap=False, extrinsic_swap=False, use_variant=False),
    "ER":     dict(image_swap=False, extrinsic_swap=True,  use_variant=True),
    "VR":     dict(image_swap=True,  extrinsic_swap=False, use_variant=True),
    "CR":     dict(image_swap=True,  extrinsic_swap=True,  use_variant=True),
}

DEFAULT_AXES = ("yaw", "pitch", "roll")
DEFAULT_MAGS = (-20, -16, -12, -8, -4, 4, 8, 12, 16, 20)


def variant_key(axis: str, mag: int) -> str:
    parts = {"yaw": 0, "pitch": 0, "roll": 0}
    parts[axis] = mag
    return f"yaw{parts['yaw']}pitch{parts['pitch']}roll{parts['roll']}"


def build_grid(axes, mags, conditions, protocols):
    grid = []
    if "Normal" in conditions:
        grid.append(dict(name="Normal", condition="Normal", variant=None,
                         axis=None, mag=None, protocol="clean",
                         target_camera=None))
    for cond in [c for c in conditions if c != "Normal"]:
        for ax, mg in product(axes, mags):
            v = variant_key(ax, mg)
            if "per_cam" in protocols:
                for cam in CAMS:
                    grid.append(dict(
                        name=f"{cond}_perCam_{cam}_{ax}{mg:+d}",
                        condition=cond, variant=v, axis=ax, mag=mg,
                        protocol="per_cam", target_camera=cam,
                    ))
            if "all_cam" in protocols:
                grid.append(dict(
                    name=f"{cond}_allCam_{ax}{mg:+d}",
                    condition=cond, variant=v, axis=ax, mag=mg,
                    protocol="all_cam", target_camera="ALL",
                ))
    return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="sedan", choices=["sedan", "suv", "bus"])
    ap.add_argument("--scene", default="scene_0220")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--conditions", nargs="+",
                    default=["Normal", "ER", "VR", "CR"],
                    choices=list(CONDITION_FLAGS))
    ap.add_argument("--protocols", nargs="+",
                    default=["per_cam", "all_cam"],
                    choices=["per_cam", "all_cam"])
    ap.add_argument("--axes", nargs="+", default=list(DEFAULT_AXES))
    ap.add_argument("--mags", nargs="+", type=int, default=list(DEFAULT_MAGS))
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = no limit; else cap output PNG count")
    ap.add_argument("--out", type=str, default=None,
                    help="Override output dir (default: <toolkit>/viz/vr/<scene>_f<frame>)")
    args = ap.parse_args()

    frame = load_frame(args.platform, args.scene, args.frame)
    meta = load_vr_metadata()
    grid = build_grid(args.axes, args.mags, args.conditions, args.protocols)
    if args.limit > 0:
        grid = grid[:args.limit]

    out_dir = (Path(args.out) if args.out
               else Path(__file__).resolve().parent / "viz" / "vr"
                    / f"{args.scene}_f{args.frame:04d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[vr-viz] {args.platform} {args.scene} frame={args.frame} → {len(grid)} configs → {out_dir}")

    for cfg in tqdm(grid, desc="VR configs"):
        flags = CONDITION_FLAGS[cfg["condition"]]
        if flags["use_variant"]:
            tgts = (CAMS if cfg["target_camera"] == "ALL"
                    else [cfg["target_camera"]])
            E_new, paths_new = apply_vr(
                frame, meta, cfg["variant"], tgts,
                image_swap=flags["image_swap"],
                extrinsic_swap=flags["extrinsic_swap"])
            hl = tgts
        else:
            E_new, paths_new = frame.extrinsics, frame.image_paths
            hl = []
        render_projection(frame, E_new, paths_new,
                          out_path=out_dir / f"{cfg['name']}.png",
                          title=cfg["name"], highlight_cams=hl)
    print(f"[vr-viz] done. {len(grid)} PNGs in {out_dir}")


if __name__ == "__main__":
    main()
