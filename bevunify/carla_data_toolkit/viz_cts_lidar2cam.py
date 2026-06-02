"""Generate lidar2cam projection viz for every CTS config.

For each (target_platform, condition) pair we render one PNG (2x3 cam grid)
with the lidar projected onto the (possibly cross-platform) image using the
(possibly cross-platform) extrinsic.

Condition matrix::
    NORMAL: img=sedan,  ext=sedan      (cross-platform baseline — IoU upper bound for target)
    EXT:    img=sedan,  ext=target     (only extrinsic differs — primary EXT axis)
    IMG:    img=target, ext=sedan      (only image differs — primary IMG axis)
    CAL:    img=target, ext=target     (= raw target run, 'with calibration')

Run::
  python -m bevunify.carla_data_toolkit.viz_cts_lidar2cam
  python -m bevunify.carla_data_toolkit.viz_cts_lidar2cam --targets suv bus \
      --scene scene_0220 --frame 0

Outputs to: bevunify/carla_data_toolkit/viz/cts/<scene>_f<frame>/<target>_<cond>.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

from tqdm import tqdm

from .projection import load_frame, build_sedan_ext_map, apply_cts
from .plot import render_projection


CONDITIONS = ["NORMAL", "EXT", "IMG", "CAL"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", nargs="+", default=["suv", "bus"],
                    choices=["sedan", "suv", "bus"])
    ap.add_argument("--scene", default="scene_0220")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS,
                    choices=CONDITIONS)
    ap.add_argument("--out", type=str, default=None,
                    help="Override output dir (default: <toolkit>/viz/cts/<scene>_f<frame>)")
    args = ap.parse_args()

    sedan_ext = build_sedan_ext_map()
    out_dir = (Path(args.out) if args.out
               else Path(__file__).resolve().parent / "viz" / "cts"
                    / f"{args.scene}_f{args.frame:04d}")
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(args.targets) * len(args.conditions)
    print(f"[cts-viz] targets={args.targets} conds={args.conditions} → {total} PNGs → {out_dir}")

    for plat in args.targets:
        frame = load_frame(plat, args.scene, args.frame)
        for cond in tqdm(args.conditions, desc=f"CTS {plat}"):
            E_new, paths_new = apply_cts(frame, cond, sedan_ext)
            render_projection(
                frame, E_new, paths_new,
                out_path=out_dir / f"{plat}_{cond}.png",
                title=f"CTS  target={plat}  condition={cond}",
                highlight_cams=None,
            )
    print(f"[cts-viz] done. {total} PNGs in {out_dir}")


if __name__ == "__main__":
    main()
