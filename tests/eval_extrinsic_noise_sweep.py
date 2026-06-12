"""6DoF extrinsic-noise robustness sweep (DSPE paper, IROS24 Fig. 3).

For each axis independently (tx/ty/tz translation, rx/ry/rz rotation), applies
zero-mean Gaussian noise of increasing std to every camera's ego->cam extrinsic
IN THE CAMERA FRAME (E' = delta @ E, matching bevunify.augmentation.ExtrinsicNoise
and the paper's "each camera's pose is perturbed independently within its own
coordinate system"), then measures IoU@0.5 (vis>=2) on the nuScenes val set.

Noise is seeded per (axis, std, batch) — DIFFERENT models evaluated with the same
arguments see IDENTICAL perturbations, making the curves directly comparable.

    $PY tests/eval_extrinsic_noise_sweep.py <experiment> <ckpt> [--dataset_dir ...]
        [--trans_stds 0.1,0.25,0.5,0.75,1.0] [--rot_stds 0.05,0.1,0.2,0.35,0.5]
        [--out results.json]

Output: per-(axis, std) IoU table + JSON.
"""
import argparse, json, math, os, torch

ap = argparse.ArgumentParser()
ap.add_argument("experiment")
ap.add_argument("ckpt")
ap.add_argument("--dataset_dir", default=os.environ.get("NUSCENES_DIR", "./data/nuscenes"))
ap.add_argument("--labels_dir", default=None)
ap.add_argument("--trans_stds", default="0.1,0.25,0.5,0.75,1.0", help="meters")
ap.add_argument("--rot_stds", default="0.05,0.1,0.2,0.35,0.5", help="radians")
ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--out", default=None, help="default: eval_results/extrin_sweep_<experiment>.json")
args = ap.parse_args()
D = args.dataset_dir
L = args.labels_dir or f"{D}/labels_aug"

import bevunify  # noqa: F401
from hydra import initialize_config_dir, compose
from bevunify.common import setup_config, setup_model_module, setup_data_module

CFG = os.path.abspath(os.path.join(os.path.dirname(bevunify.__file__), "..", "config"))
with initialize_config_dir(version_base="1.3", config_dir=CFG):
    cfg = compose(config_name="config", overrides=[f"+experiment={args.experiment}",
        f"data.dataset_dir={D}", f"data.labels_dir={L}",
        "experiment.logger=csv", "experiment.save_dir=/tmp/eaf_eval/",
        "loader.val_batch_size=12", "loader.num_workers=8", "trainer.devices=1"])
    setup_config(cfg)
    mm = setup_model_module(cfg)
    dm = setup_data_module(cfg)
sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)
miss, unexp = mm.load_state_dict(sd.get("state_dict", sd), strict=False)
print(f"[{args.experiment}] ckpt {args.ckpt} (missing {len(miss)}, unexpected {len(unexp)})")
mm = mm.cuda().eval()


def _axis_delta(axis, std, gen, n):
    """(n,4,4) camera-frame deltas: Gaussian noise on ONE axis. axis in tx,ty,tz,rx,ry,rz."""
    deltas = torch.eye(4).repeat(n, 1, 1)
    vals = torch.randn(n, generator=gen) * std
    if axis.startswith("t"):
        deltas[:, "xyz".index(axis[1]), 3] = vals
    else:
        i = "xyz".index(axis[1])
        for k in range(n):
            a = float(vals[k]); c, s = math.cos(a), math.sin(a)
            j, l = [(1, 2), (0, 2), (0, 1)][i]
            Rm = torch.eye(3)
            Rm[j, j], Rm[l, l] = c, c
            sign = -1.0 if i == 1 else 1.0       # standard right-handed rotations
            Rm[j, l], Rm[l, j] = -s * sign, s * sign
            deltas[k, :3, :3] = Rm
    return deltas


@torch.no_grad()
def run_eval(axis=None, std=0.0):
    tp = fp = fn = 0
    for bidx, batch in enumerate(dm.val_dataloader()):
        bc = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        if axis is not None and std > 0:
            E = bc["extrinsics"]                                  # (B,N,4,4) ego->cam
            B, N = E.shape[:2]
            # deterministic per (axis, std, batch index) -> identical across models
            gen = torch.Generator().manual_seed(hash((axis, round(std, 4), bidx)) & 0x7FFFFFFF)
            deltas = _axis_delta(axis, std, gen, B * N).to(E.device)
            bc["extrinsics"] = (deltas.view(B, N, 4, 4) @ E)
        p = mm.backbone(bc)["vehicle"].sigmoid()
        t = batch["vehicle"].bool().cuda()
        m = (batch["vehicle_visibility"] >= 2).cuda()[:, None].expand_as(p)
        pv = p[m] >= args.threshold
        tv = t[m]
        tp += (pv & tv).sum().item(); fp += (pv & ~tv).sum().item(); fn += (~pv & tv).sum().item()
    return tp / (tp + fp + fn + 1e-7)


results = {"experiment": args.experiment, "ckpt": args.ckpt, "threshold": args.threshold, "curves": {}}
clean = run_eval()
results["clean_iou"] = clean
print(f"[{args.experiment}] clean IoU@{args.threshold:g}_vis2 = {clean:.4f}")
trans = [float(x) for x in args.trans_stds.split(",")]
rot = [float(x) for x in args.rot_stds.split(",")]
for axis, stds, unit in [("tx", trans, "m"), ("ty", trans, "m"), ("tz", trans, "m"),
                         ("rx", rot, "rad"), ("ry", rot, "rad"), ("rz", rot, "rad")]:
    curve = [(0.0, clean)]
    for s in stds:
        iou = run_eval(axis, s)
        curve.append((s, iou))
        print(f"  [{axis}] std={s:g}{unit}: IoU={iou:.4f}")
    results["curves"][axis] = curve

out = args.out or f"eval_results/extrin_sweep_{args.experiment}.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {out}")
