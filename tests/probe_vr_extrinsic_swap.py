"""Empirically verify VR/ER/CR extrinsic swap correctness.

For scene_0220 frame 0 of carla sedan_eval, runs LoadDataTransform under four
configurations:

  Normal              image_swap=False, extrinsic_swap=False  (no perturbation)
  ER_yaw20            image_swap=False, extrinsic_swap=True
  VR_yaw20            image_swap=True,  extrinsic_swap=False   (PRIMARY VR mode)
  CR_yaw20            image_swap=True,  extrinsic_swap=True    (= ER + VR images)

then verifies:
  1. Normal vs ER:  CAM_FRONT extrinsic CHANGED (Frobenius > 1e-4)
  2. Normal vs VR:  CAM_FRONT extrinsic UNCHANGED (Frobenius ≈ 0)
  3. Normal vs CR:  CAM_FRONT extrinsic CHANGED
  4. ER  vs CR:     CAM_FRONT extrinsic MATCH  (both apply same correction)
  5. ER produces E_actual == E_expected, where
        E_expected = (cam_variant_from_egocam @ egocam_from_cam_baseline) @ E_baseline
     (independently recomputed from viewpoint_metadata.json).
  6. eval_target_cameras={"CAM_FRONT"} only mutates CAM_FRONT — CAM_BACK stays at
     baseline.

Run:
  python /home/hanyan_arch/git/BEV_seg/bevunify/tests/probe_vr_extrinsic_swap.py
"""
import json
import pathlib
import sys

import numpy as np
import torch
from pyquaternion import Quaternion

# Allow direct execution without bevunify install.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bevunify.carla_data import LoadDataTransform, Sample  # noqa: E402

# ── paths (carla_geobev sedan_eval) ──────────────────────────────────────────
DATASET_DIR    = "/home/hanyan_arch/data/carla_geobev"
LABELS_DIR     = "/home/hanyan_arch/data/carla_geobev_labels/gaussianlss/sedan_eval"
VP_META_PATH   = "/NHNHOME/WORKSPACE/0526040099_A/jeongtae/carla_VR/viewpoint_metadata.json"
SCENE          = "scene_0220"
FRAME_IDX      = 0          # frame-0000 → vr_frame = 0 (no path bug for this idx)
VARIANT        = "yaw20pitch0roll0"
CAM_FRONT_IDX  = 1          # cam_channels = [CAM_FRONT_LEFT, CAM_FRONT, ...]
CAM_BACK_IDX   = 4
OUT_TXT        = pathlib.Path("/tmp/probe_vr_extrinsic_swap.txt")

IMG_CFG = {"h": 224, "w": 480, "top_crop": 46}


def make_xform(image_swap, extrinsic_swap, target_cameras=None, variant=VARIANT):
    return LoadDataTransform(
        dataset_dir=DATASET_DIR,
        labels_dir=LABELS_DIR,
        image_config=IMG_CFG,
        num_classes=12,
        augment="none",
        split_intrin_extrin=True,
        label_indices=[[4, 5, 6, 7, 8, 10, 11]],
        eval_viewpoint_variant=variant,
        viewpoint_metadata_path=VP_META_PATH,
        eval_image_swap=image_swap,
        eval_extrinsic_swap=extrinsic_swap,
        eval_target_cameras=target_cameras,
    )


def load_sample(xform):
    samples = json.loads((pathlib.Path(LABELS_DIR) / f"{SCENE}.json").read_text())
    s = Sample(**samples[FRAME_IDX])
    return xform(s)


def fro(a, b):
    return float(torch.linalg.norm((a - b).float()))


def expected_E_variant(scene, cam, variant, E_baseline_np):
    """Recompute correction independently from raw quaternions in viewpoint_metadata."""
    meta = json.loads(pathlib.Path(VP_META_PATH).read_text())
    base = meta["scenes"][scene][cam]["yaw0pitch0roll0"]
    var  = meta["scenes"][scene][cam][variant]

    def egocam_from_cam(s2e):
        R = Quaternion(s2e["sensor2ego_rotation"]).rotation_matrix
        t = np.asarray(s2e["sensor2ego_translation"])
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
        return T

    def cam_from_egocam(s2e):
        R = Quaternion(s2e["sensor2ego_rotation"]).rotation_matrix
        t = np.asarray(s2e["sensor2ego_translation"])
        T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ t
        return T

    correction = cam_from_egocam(var) @ egocam_from_cam(base)
    return (correction.astype(np.float32) @ E_baseline_np).astype(np.float32)


def header(s):
    bar = "=" * 70
    return f"\n{bar}\n {s}\n{bar}"


def main():
    log_lines = []
    def emit(s=""):
        print(s)
        log_lines.append(str(s))

    emit(header("Loading samples"))

    # Variant=None disables both swap pathways (true Normal baseline).
    normal_x = LoadDataTransform(
        dataset_dir=DATASET_DIR, labels_dir=LABELS_DIR, image_config=IMG_CFG,
        num_classes=12, augment="none", split_intrin_extrin=True,
        label_indices=[[4, 5, 6, 7, 8, 10, 11]],
        eval_viewpoint_variant=None,        # no perturbation
        viewpoint_metadata_path=VP_META_PATH,
    )
    er_x     = make_xform(image_swap=False, extrinsic_swap=True)
    vr_x     = make_xform(image_swap=True,  extrinsic_swap=False)
    cr_x     = make_xform(image_swap=True,  extrinsic_swap=True)

    # VR/CR open VR images on disk; that's wanted, but we want extrinsic numbers
    # only — sample-level success of image load is incidental.
    normal = load_sample(normal_x)
    er     = load_sample(er_x)
    vr     = load_sample(vr_x)
    cr     = load_sample(cr_x)

    cam_channels = json.loads(
        (pathlib.Path(LABELS_DIR) / f"{SCENE}.json").read_text()
    )[FRAME_IDX]["cam_channels"]
    emit(f"cam_channels: {cam_channels}")
    assert cam_channels[CAM_FRONT_IDX] == "CAM_FRONT", cam_channels
    assert cam_channels[CAM_BACK_IDX]  == "CAM_BACK",  cam_channels

    E_norm_f = normal["extrinsics"][CAM_FRONT_IDX]
    E_er_f   = er["extrinsics"][CAM_FRONT_IDX]
    E_vr_f   = vr["extrinsics"][CAM_FRONT_IDX]
    E_cr_f   = cr["extrinsics"][CAM_FRONT_IDX]

    emit(header("CAM_FRONT extrinsics"))
    emit(f"Normal E:\n{E_norm_f.numpy()}")
    emit(f"\nER     E:\n{E_er_f.numpy()}")
    emit(f"\nVR     E:\n{E_vr_f.numpy()}")
    emit(f"\nCR     E:\n{E_cr_f.numpy()}")

    # ── Assertions ────────────────────────────────────────────────────────────
    results = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        line = f"[{status}] {name}  {detail}"
        emit(line)
        results.append((name, condition))

    emit(header("Assertions"))

    d_norm_er = fro(E_norm_f, E_er_f)
    d_norm_vr = fro(E_norm_f, E_vr_f)
    d_norm_cr = fro(E_norm_f, E_cr_f)
    d_er_cr   = fro(E_er_f,   E_cr_f)

    check("Normal vs ER  : CAM_FRONT E CHANGED  (>1e-4)",
          d_norm_er > 1e-4, f"|Δ|_F = {d_norm_er:.6e}")
    check("Normal vs VR  : CAM_FRONT E UNCHANGED (<1e-6)",
          d_norm_vr < 1e-6, f"|Δ|_F = {d_norm_vr:.6e}")
    check("Normal vs CR  : CAM_FRONT E CHANGED  (>1e-4)",
          d_norm_cr > 1e-4, f"|Δ|_F = {d_norm_cr:.6e}")
    check("ER     vs CR  : CAM_FRONT E MATCH    (<1e-6)",
          d_er_cr   < 1e-6, f"|Δ|_F = {d_er_cr:.6e}")

    # ── Math-equivalence check ────────────────────────────────────────────────
    emit(header("Independent math verification (E_expected vs E_actual under ER)"))
    E_base_np = E_norm_f.numpy()
    E_expected = expected_E_variant(SCENE, "CAM_FRONT", VARIANT, E_base_np)
    diff = float(np.linalg.norm(E_expected - E_er_f.numpy()))
    emit(f"E_expected =\n{E_expected}")
    emit(f"|E_expected - E_actual_ER|_F = {diff:.6e}")
    check("E_expected matches ER's CAM_FRONT extrinsic (<1e-5)",
          diff < 1e-5, f"|Δ|_F = {diff:.6e}")

    # ── target_cameras gating ─────────────────────────────────────────────────
    emit(header("target_cameras={CAM_FRONT} gating check"))
    gated_x = make_xform(image_swap=False, extrinsic_swap=True,
                         target_cameras={"CAM_FRONT"})
    gated = load_sample(gated_x)
    E_g_front = gated["extrinsics"][CAM_FRONT_IDX]
    E_g_back  = gated["extrinsics"][CAM_BACK_IDX]
    E_norm_back = normal["extrinsics"][CAM_BACK_IDX]

    d_g_front_vs_norm = fro(E_g_front, E_norm_f)
    d_g_back_vs_norm  = fro(E_g_back,  E_norm_back)
    d_g_front_vs_er   = fro(E_g_front, E_er_f)
    emit(f"gated CAM_FRONT vs Normal CAM_FRONT  |Δ|_F = {d_g_front_vs_norm:.6e}")
    emit(f"gated CAM_BACK  vs Normal CAM_BACK   |Δ|_F = {d_g_back_vs_norm:.6e}")
    emit(f"gated CAM_FRONT vs ER    CAM_FRONT   |Δ|_F = {d_g_front_vs_er:.6e}")

    check("Gated CAM_FRONT CHANGED   (matches ER, >1e-4)",
          d_g_front_vs_norm > 1e-4)
    check("Gated CAM_BACK  UNCHANGED (<1e-6)",
          d_g_back_vs_norm  < 1e-6)
    check("Gated CAM_FRONT MATCHES ungated-ER CAM_FRONT (<1e-6)",
          d_g_front_vs_er   < 1e-6)

    # ── summary ───────────────────────────────────────────────────────────────
    emit(header("Summary"))
    fails = [n for n, ok in results if not ok]
    emit(f"{len(results) - len(fails)}/{len(results)} assertions PASSED")
    if fails:
        emit(f"FAILED: {fails}")
    else:
        emit("ALL ASSERTIONS PASSED.")

    OUT_TXT.write_text("\n".join(log_lines) + "\n")
    emit(f"\nLog written to {OUT_TXT}")

    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
