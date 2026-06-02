"""Throwaway test v2: extract the *model-facing* geometry tensor for each wrapper,
decode back to ego-frame (pos, fwd) per camera, and visualize Normal vs ER.

This is the deeper version of v1: v1 only checked `batch["extrinsics"]` (loader output);
v2 reproduces what each wrapper actually *hands to* the underlying model — so a wrong
inversion or wrong reference-frame composition inside a wrapper would surface here.

Mapping per model (matches wrapper source):
  cvt          : net(batch)                                  -> ego->cam     (batch["extrinsics"])
  gaussianlss  : host; consumes batch["extrinsics"] directly -> ego->cam
  lss          : net(... rots, trans ...)                    -> cam->ego     (rots_trans(batch))
  lara         : net(... rots, trans ...)                    -> cam->ego     (rots_trans(batch))
  pointbev     : net(... rots, trans ...)                    -> cam->ego     (rots_trans(batch))
  simplebev    : net(... cam0_T_camXs ...)                   -> CAM_FRONT-relative

Decode each to (pos_in_ego, fwd_in_ego) for cross-model comparison.
"""
import json, sys, numpy as np, torch
from pathlib import Path

REPO = Path("/home/hanyan_arch/git/BEV_seg/bevunify")
for p in (str(REPO / "third_party/GaussianLSS"), str(REPO)):
    if p not in sys.path: sys.path.insert(0, p)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bevunify  # noqa
from bevunify.data_toggle import get_data_carla
from bevunify.common import setup_config
from bevunify.wrappers.geom import rots_trans
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

MODELS = ["cvt", "gaussianlss", "lara", "lss", "pointbev", "simplebev"]
SCENE, FRAME = "scene_0220", 0
ER = "yaw-20pitch0roll0"
CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"]
CCOL = {"CAM_FRONT_LEFT":"#5cb338","CAM_FRONT":"#ff5e5b","CAM_FRONT_RIGHT":"#fbb13c",
        "CAM_BACK_LEFT":"#268bd2","CAM_BACK":"#6c71c4","CAM_BACK_RIGHT":"#b58900"}
ARROW_LEN = 1.5  # m — long enough to actually see the 20° rotation


def compose_cfg(m):
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(REPO/"config")):
        cfg = compose(config_name="config", overrides=[
            f"+experiment={m}", "data=carla", "data.version=v1.0-carla_sedan",
            "data.split_intrin_extrin=True",
            "loader.val_batch_size=1", "loader.num_workers=0",
            "experiment.save_dir=/tmp/bevunify_probe/", "experiment.logger=csv"])
        setup_config(cfg)
    return cfg


def get_ds(cfg):
    d = dict(cfg.data); d.pop("dataset", None)
    for ds in get_data_carla(split="val", **d):
        if ds.scene_name == SCENE: return ds


def set_normal(t):
    t.eval_viewpoint_variant = None
    t.eval_image_swap = False
    t.eval_extrinsic_swap = False
    t.eval_target_cameras = None


def set_er(t, cfg):
    t.eval_viewpoint_variant = ER
    t.eval_image_swap = False
    t.eval_extrinsic_swap = True
    t.eval_target_cameras = {"CAM_FRONT"}
    if t.viewpoint_metadata is None:
        mp = t.viewpoint_metadata_path or cfg.data.get("viewpoint_metadata_path")
        t.viewpoint_metadata = json.loads(Path(mp).read_text())
        t.vr_root = Path(t.viewpoint_metadata["vr_root"])


def make_batch(sample):
    """Add a B=1 dim to every tensor (mirror minimum collate the wrappers expect)."""
    out = {}
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)
        else:
            out[k] = v
    return out


def decode_ego_to_cam(E):
    """E: (6,4,4) ego->cam. Returns list of (pos_ego, fwd_ego) per cam."""
    inv = np.linalg.inv(E)
    return [(inv[k, :3, 3], inv[k, :3, :3] @ np.array([0, 0, 1.])) for k in range(6)]


def decode_cam_to_ego(R, t):
    """R: (6,3,3) cam->ego, t: (6,3). Returns list of (pos_ego, fwd_ego)."""
    return [(t[k], R[k] @ np.array([0, 0, 1.])) for k in range(6)]


def decode_cam0_relative(cam0_T_camXs, E0):
    """cam0_T_camXs: (6,4,4) transforming a point from cam_k frame to cam_0 (CAM_FRONT) frame.
    Re-lift to ego using E0 (ego->cam_0), which the wrapper itself reads from the batch."""
    E0_inv = np.linalg.inv(E0)  # cam_0 -> ego
    poses = []
    for k in range(6):
        pos_c0 = cam0_T_camXs[k, :3, 3]                       # cam_k origin in cam_0 frame
        fwd_c0 = cam0_T_camXs[k, :3, :3] @ np.array([0, 0, 1.])
        pos_e = E0_inv[:3, :3] @ pos_c0 + E0_inv[:3, 3]       # lift to ego
        fwd_e = E0_inv[:3, :3] @ fwd_c0
        poses.append((pos_e, fwd_e))
    return poses


def model_facing_geometry(model_name, batch):
    """Reproduce the *exact* geometric tensor each wrapper hands to its underlying model,
    then decode back to ego frame for comparison."""
    if model_name in ("cvt", "gaussianlss"):
        E = batch["extrinsics"][0].cpu().numpy()
        return decode_ego_to_cam(E), "batch.extrinsics  (ego->cam)"
    elif model_name in ("lss", "lara", "pointbev"):
        R, t = rots_trans(batch)
        return decode_cam_to_ego(R[0].cpu().numpy(), t[0].cpu().numpy()), "rots_trans()  (cam->ego)"
    elif model_name == "simplebev":
        E = batch["extrinsics"][0].cpu().numpy()              # (6,4,4)
        E0 = E[1]                                              # CAM_FRONT
        cam0_T_camXs = E0 @ np.linalg.inv(E)                   # (6,4,4)
        return decode_cam0_relative(cam0_T_camXs, E0), "cam0_T_camXs  (CAM_FRONT-relative)"
    else:
        raise ValueError(model_name)


def run(m):
    cfg = compose_cfg(m); ds = get_ds(cfg); t = ds.transform
    set_normal(t)
    bn = make_batch(ds[FRAME])
    pn, repr_n = model_facing_geometry(m, bn)
    set_er(t, cfg)
    be = make_batch(ds[FRAME])
    pe, repr_e = model_facing_geometry(m, be)
    set_normal(t)
    assert repr_n == repr_e
    return pn, pe, repr_n


def draw(ax, pn, pe, title):
    ax.set_aspect("equal"); ax.set_xlim(-3, 3); ax.set_ylim(-3, 5)
    ax.axhline(0, color="lightgray", lw=0.4)
    ax.axvline(0, color="lightgray", lw=0.4)
    # ego marker + forward indicator
    ax.plot(0, 0, "k+", markersize=14, mew=2)
    ax.annotate("ego", xy=(0.18, 0.18), fontsize=8)
    for cam, (pn1, fn1), (pe1, fe1) in zip(CAMS, pn, pe):
        c = CCOL[cam]
        # Normal: solid circle + solid arrow
        ax.plot(pn1[1], pn1[0], "o", color=c, markersize=7, zorder=3, mec="black", mew=0.5)
        ax.annotate("", xy=(pn1[1] + ARROW_LEN * fn1[1], pn1[0] + ARROW_LEN * fn1[0]),
                    xytext=(pn1[1], pn1[0]),
                    arrowprops=dict(arrowstyle="->", color=c, lw=2.0))
        # ER: only draw if it actually differs (skip cams that weren't perturbed)
        moved = not (np.allclose(pn1, pe1, atol=1e-6) and np.allclose(fn1, fe1, atol=1e-6))
        if moved:
            # nudge the ER marker visibly off the Normal one so they don't overlap
            ax.plot(pe1[1] + 0.05, pe1[0] + 0.05, "x", color=c,
                    markersize=11, mew=2.2, zorder=4)
            ax.annotate("", xy=(pe1[1] + ARROW_LEN * fe1[1], pe1[0] + ARROW_LEN * fe1[0]),
                        xytext=(pe1[1], pe1[0]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.8, ls=(0, (4, 2))))
        # cam label (offset so it doesn't sit on the arrow)
        ax.annotate(cam.replace("CAM_", ""), xy=(pn1[1] - 0.5, pn1[0] - 0.35),
                    fontsize=6.5, color=c)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("ego.y left+ [m]", fontsize=7)
    ax.set_ylabel("ego.x fwd+ [m]", fontsize=7)
    ax.tick_params(labelsize=6)


if __name__ == "__main__":
    results = {}
    print(f"scene={SCENE} frame={FRAME} ER={ER} target=CAM_FRONT\n")
    print("Each model: decode the *model-facing* geometry tensor back to ego frame.\n")
    for m in MODELS:
        print(f"  {m} ...", end=" ", flush=True)
        try:
            pn, pe, repr_str = run(m)
            results[m] = (pn, pe, repr_str)
            print(f"ok  [{repr_str}]")
        except Exception as e:
            print(f"ERR: {type(e).__name__}: {e}")
            results[m] = None

    # 1) Cross-model consistency on CAM_FRONT pose in Normal config
    print("\n=== cross-model consistency (Normal, CAM_FRONT pos in ego) ===")
    ref = None
    for m in MODELS:
        if results.get(m) is None: continue
        pos = results[m][0][CAMS.index("CAM_FRONT")][0]
        if ref is None:
            ref = pos; tag = "(ref)"
        else:
            tag = f"|Δ vs cvt| = {np.linalg.norm(pos - ref):.2e}"
        print(f"  {m:>12}  pos = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})  {tag}")

    # 2) Cross-model consistency on ALL 6 cams (max position discrepancy)
    print("\n=== cross-model consistency (Normal, max |Δpos| over 6 cams vs cvt) ===")
    if results.get("cvt"):
        ref_all = np.stack([results["cvt"][0][k][0] for k in range(6)])  # (6,3)
        for m in MODELS:
            if results.get(m) is None: continue
            pos_all = np.stack([results[m][0][k][0] for k in range(6)])
            diffs = np.linalg.norm(pos_all - ref_all, axis=1)
            print(f"  {m:>12}  max |Δ| = {diffs.max():.2e}  mean = {diffs.mean():.2e}")

    # 3) ER perturbation: CAM_FRONT forward rotation
    print("\n=== ER perturbation: CAM_FRONT Δyaw (expect ~+20° = inverse of yaw=-20°) ===")
    for m in MODELS:
        if results.get(m) is None: continue
        idx = CAMS.index("CAM_FRONT")
        fn = results[m][0][idx][1][:2]
        fe = results[m][1][idx][1][:2]
        nn = fn / max(np.linalg.norm(fn), 1e-9)
        ne = fe / max(np.linalg.norm(fe), 1e-9)
        ang = np.degrees(np.arctan2(nn[0] * ne[1] - nn[1] * ne[0], nn @ ne))
        st = "OK" if abs(abs(ang) - 20) < 1 else "WRONG"
        print(f"  {m:>12}  Δyaw = {ang:+6.2f}°  [{st}]")

    # 4) ER perturbation should NOT move the OTHER 5 cams
    print("\n=== ER non-target cams: max |Δpos| (should be ~0) ===")
    for m in MODELS:
        if results.get(m) is None: continue
        deltas = []
        for k, cam in enumerate(CAMS):
            if cam == "CAM_FRONT": continue
            pn1 = results[m][0][k][0]; pe1 = results[m][1][k][0]
            deltas.append(np.linalg.norm(pn1 - pe1))
        print(f"  {m:>12}  max |Δ| over 5 non-target = {max(deltas):.2e}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    for ax, m in zip(axes.flat, MODELS):
        if results.get(m) is None:
            ax.set_title(f"{m}: ERR"); ax.axis("off"); continue
        pn, pe, repr_str = results[m]
        draw(ax, pn, pe, f"{m}\n{repr_str}\n• Normal  |  × ER (target=CAM_FRONT)")
    fig.suptitle(
        f"per-wrapper model-facing geometry, decoded to ego frame  |  ER {ER}",
        fontsize=12)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "viz" / "test_per_model_extrinsic.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"\nsaved: {out}")
