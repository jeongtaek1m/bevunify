"""Empirical probe for CTS image+extrinsic swap.

Loads suv_eval scene_0220 frame 0 four ways (NORMAL/EXT/IMG/CAL) using the
real bevunify.carla_data.LoadDataTransform with the same CTS hooks that
bevunify.eval._mutate_cts sets (eval.py:190-201), and verifies that the
right tensor flips when the right flag is toggled. Citing line numbers
relative to bevunify/carla_data.py:

  - cts_img_to_sedan branch       carla_data.py:208-209  (_cts_path_to_sedan)
  - cts_ext_override branch       carla_data.py:240-243

If image stays sedan/target by accident or extrinsic doesn't actually swap,
the 631-config VR + CTS eval will produce wrong numbers silently. This script
is the cheap pre-flight.
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image

# Import directly from the package, not the toggle subclass — we want to
# exercise the raw CTS code path in carla_data.LoadDataTransform.
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bevunify.carla_data import LoadDataTransform, Sample  # noqa: E402

# ── constants pulled from real defaults ───────────────────────────────────────

DATASET_DIR = "/home/hanyan_arch/data/carla_geobev"
LABELS_BASE = pathlib.Path("/home/hanyan_arch/data/carla_geobev_labels/gaussianlss")
SUV_LABELS = LABELS_BASE / "suv_eval"
SEDAN_LABELS = LABELS_BASE / "sedan_eval"

# image_config from config/data/carla.yaml:31-34
IMAGE_CONFIG = dict(h=224, w=480, top_crop=46)
NUM_CLASSES = 12
CAM = "CAM_FRONT"

# bookkeeping
OUT_PNG = pathlib.Path("/tmp/probe_cts_swap.png")
OUT_TXT = pathlib.Path("/tmp/probe_cts_swap.txt")


# ── helpers ──────────────────────────────────────────────────────────────────

def first_sample_of(scene_json: pathlib.Path) -> dict:
    return json.loads(scene_json.read_text())[0]


def build_sedan_ext_map(sedan_labels_dir: pathlib.Path,
                        target_labels_dir: pathlib.Path) -> dict[str, np.ndarray]:
    """Mirror eval._build_sedan_ext_delta: per-cam rig delta
    D_ch = E_sedan(0) @ inv(E_target(0)). Applied at load time as
    E_sedan(t) = D_ch @ E_target(t) — extrinsics wobble per frame (ego
    suspension); the rig delta is frame-invariant so this reconstructs the
    per-frame sedan extrinsic exactly."""
    def _frame0(p):
        first = next(iter(sorted(p.glob("scene_*.json"))), None)
        assert first is not None, f"no scene_*.json under {p}"
        s = json.loads(first.read_text())[0]
        return {ch: np.array(e, dtype=np.float64)
                for ch, e in zip(s["cam_channels"], s["extrinsics"])}, first.name

    sed, src = _frame0(sedan_labels_dir)
    tgt, _ = _frame0(target_labels_dir)
    delta = {ch: (sed[ch] @ np.linalg.inv(tgt[ch])).astype(np.float32)
             for ch in sed if ch in tgt}
    return delta, src


def make_transform() -> LoadDataTransform:
    """Match the defaults used at eval time for carla."""
    return LoadDataTransform(
        dataset_dir=DATASET_DIR,
        labels_dir=str(SUV_LABELS),
        image_config=IMAGE_CONFIG,
        num_classes=NUM_CLASSES,
        augment="none",
        split_intrin_extrin=True,        # carla.yaml:47
        label_indices=None,
    )


def apply_cts(t: LoadDataTransform, img_to_sedan: bool, ext_override: dict | None,
              target_platform: str) -> None:
    """Same mutation as eval._mutate_cts (eval.py:190-201)."""
    t.cts_platform = target_platform
    t.cts_img_to_sedan = bool(img_to_sedan)
    t.cts_ext_override = ext_override
    # VR hooks off
    t.eval_viewpoint_variant = None
    t.eval_image_swap = False
    t.eval_extrinsic_swap = False
    t.eval_target_cameras = None


def load_one(t: LoadDataTransform, raw_sample: dict) -> dict[str, Any]:
    sample = Sample(**raw_sample)
    out = t.get_cameras(sample, **IMAGE_CONFIG)
    cam_channels = raw_sample["cam_channels"]
    k = cam_channels.index(CAM)
    return dict(
        image=out["image"][k].clone(),
        extrinsic=out["extrinsics"][k].clone(),
        cam_idx=k,
    )


def tensors_equal(a: torch.Tensor, b: torch.Tensor, atol: float = 0.0) -> bool:
    return a.shape == b.shape and torch.allclose(a, b, atol=atol)


# ── main probe ───────────────────────────────────────────────────────────────

def main() -> int:
    findings: list[str] = []
    pass_fail: list[tuple[str, bool, str]] = []

    def check(label: str, ok: bool, detail: str = "") -> bool:
        pass_fail.append((label, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(': ' + detail) if detail else ''}")
        return ok

    # 1. Load suv_eval scene_0220 frame 0
    suv_json = SUV_LABELS / "scene_0220.json"
    assert suv_json.exists(), f"missing {suv_json}"
    raw = first_sample_of(suv_json)
    print(f"[probe] suv_eval scene_0220 frame 0; cam_channels = {raw['cam_channels']}")
    print(f"[probe] front image path (target) = {raw['images'][raw['cam_channels'].index(CAM)]}")

    # 2. Build sedan rig-delta map (D_ch = E_sedan(0) @ inv(E_suv(0)))
    sedan_map, sedan_src = build_sedan_ext_map(SEDAN_LABELS, SUV_LABELS)
    print(f"[probe] sedan rig delta built from {sedan_src}; cams = {sorted(sedan_map)}")

    # 3. Pre-check: cts_path_to_sedan should resolve to a real file on disk
    t_probe = make_transform()
    apply_cts(t_probe, img_to_sedan=True, ext_override=None, target_platform="suv")
    front_idx = raw["cam_channels"].index(CAM)
    swapped_rel = t_probe._cts_path_to_sedan(raw["images"][front_idx], CAM)
    swapped_abs = pathlib.Path(DATASET_DIR) / swapped_rel
    print(f"[probe] swapped path = {swapped_rel}")
    try:
        Image.open(swapped_abs).verify()
        on_disk_ok = True
        msg = f"opened {swapped_abs}"
    except Exception as e:
        on_disk_ok = False
        msg = f"{type(e).__name__}: {e} (path={swapped_abs})"
    if not check("cts_path_to_sedan resolves to readable file [BLOCKER if fail]",
                 on_disk_ok, msg):
        findings.append("BLOCKER: cts_path_to_sedan path does not exist on disk")

    # 4. Load 4 conditions
    print("\n[probe] loading 4 conditions...")
    results: dict[str, dict[str, Any]] = {}
    cts_specs = {
        "NORMAL": dict(cts_img_to_sedan=True,  ext_override=sedan_map),  # both sedan
        "EXT":    dict(cts_img_to_sedan=True,  ext_override=None),       # sedan img + target ext
        "IMG":    dict(cts_img_to_sedan=False, ext_override=sedan_map),  # target img + sedan ext (PRIMARY)
        "CAL":    dict(cts_img_to_sedan=False, ext_override=None),       # both target
    }
    for cond, spec in cts_specs.items():
        t = make_transform()
        apply_cts(t, img_to_sedan=spec["cts_img_to_sedan"],
                  ext_override=spec["ext_override"], target_platform="suv")
        results[cond] = load_one(t, raw)
        print(f"  loaded {cond}: image.sum={results[cond]['image'].sum():.2f} "
              f"ext[0,3]={results[cond]['extrinsic'][0, 3].item():+.5f}")

    print("\n[probe] === assertions ===")

    # 5a. NORMAL.image == EXT.image  (both load sedan RGB)
    check("NORMAL.image == EXT.image (both sedan img)",
          tensors_equal(results["NORMAL"]["image"], results["EXT"]["image"]),
          detail=f"max|delta|={(results['NORMAL']['image']-results['EXT']['image']).abs().max().item():.4e}")

    # 5b. NORMAL.image != IMG.image (sedan vs suv)
    delta = (results["NORMAL"]["image"] - results["IMG"]["image"]).abs().max().item()
    check("NORMAL.image != IMG.image (sedan vs target img)",
          delta > 1e-3,
          detail=f"max|delta|={delta:.4e}")

    # 5c. NORMAL.extrinsic == IMG.extrinsic (both sedan ext)
    de = (results["NORMAL"]["extrinsic"] - results["IMG"]["extrinsic"]).abs().max().item()
    check("NORMAL.extrinsic == IMG.extrinsic (both sedan ext)",
          tensors_equal(results["NORMAL"]["extrinsic"], results["IMG"]["extrinsic"]),
          detail=f"max|delta|={de:.4e}")

    # 5d. NORMAL.extrinsic != EXT.extrinsic (sedan vs target)
    de = (results["NORMAL"]["extrinsic"] - results["EXT"]["extrinsic"]).abs().max().item()
    check("NORMAL.extrinsic != EXT.extrinsic (sedan vs target ext)",
          de > 1e-4,
          detail=f"max|delta|={de:.4e}")

    # 5e. All 4 mutually distinct on the (image, extrinsic) pair
    keys = ["NORMAL", "EXT", "IMG", "CAL"]
    collapse = False
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            same_img = tensors_equal(results[a]["image"], results[b]["image"])
            same_ext = tensors_equal(results[a]["extrinsic"], results[b]["extrinsic"])
            if same_img and same_ext:
                collapse = True
                print(f"  collapse: {a} == {b} on both image and extrinsic")
    check("All 4 conditions mutually distinct on (image, extrinsic)", not collapse)

    # Additional sanity: EXT.image == IMG ? Should be FALSE (different image source).
    same = tensors_equal(results["EXT"]["image"], results["IMG"]["image"])
    check("EXT.image != IMG.image (sedan vs target img, sanity)", not same)

    # 6. Verify the rig delta reconstructs the PER-FRAME sedan extrinsic:
    #    D[CAM] @ E_suv(frame) must equal E_sedan(frame) for the probed frame.
    #    (For scene_0220 frame 0 — the delta's own build frame — this is exact
    #    by construction; tolerance covers float32 label precision.)
    sedan_raw = first_sample_of(SEDAN_LABELS / "scene_0220.json")
    sedan_idx = sedan_raw["cam_channels"].index(CAM)
    sedan_E_in_scene = np.array(sedan_raw["extrinsics"][sedan_idx], dtype=np.float32)
    suv_E_in_scene = np.array(raw["extrinsics"][front_idx], dtype=np.float32)
    reconstructed = sedan_map[CAM] @ suv_E_in_scene
    de = float(np.abs(sedan_E_in_scene - reconstructed).max())
    check("delta[CAM_FRONT] @ E_suv(frame) == E_sedan(frame) (per-frame reconstruction)",
          de < 1e-3,
          detail=f"max|delta|={de:.4e}")
    # And the loader must actually produce that reconstructed extrinsic in IMG.
    de_loader = float((results["IMG"]["extrinsic"] - torch.tensor(reconstructed)).abs().max().item())
    check("IMG.extrinsic == delta @ raw target extrinsic (the loader applied the delta)",
          de_loader < 1e-5,
          detail=f"max|delta|={de_loader:.4e}")

    # Also: CAL.extrinsic should equal the raw target extrinsic.
    target_E = torch.tensor(np.array(raw["extrinsics"][front_idx], dtype=np.float32))
    de_target = float((results["CAL"]["extrinsic"] - target_E).abs().max().item())
    check("CAL.extrinsic == raw target extrinsic (no override path)",
          de_target < 1e-6,
          detail=f"max|delta|={de_target:.4e}")
    # And EXT.extrinsic should also equal target.
    de_ext_target = float((results["EXT"]["extrinsic"] - target_E).abs().max().item())
    check("EXT.extrinsic == raw target extrinsic (override=None)",
          de_ext_target < 1e-6,
          detail=f"max|delta|={de_ext_target:.4e}")

    # 7. Side-by-side montage + extrinsic dump
    print("\n[probe] writing artifacts...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        for ax, cond in zip(axes, ["NORMAL", "EXT", "IMG", "CAL"]):
            img = results[cond]["image"].permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f"{cond}\nE[0,3]={results[cond]['extrinsic'][0, 3]:+.4f}", fontsize=10)
            ax.axis("off")
        fig.suptitle("CTS swap probe — suv_eval scene_0220 frame 0 CAM_FRONT", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=80)
        plt.close(fig)
        print(f"  wrote {OUT_PNG}")
    except Exception as e:
        print(f"  WARN: failed to write {OUT_PNG}: {e}")

    with OUT_TXT.open("w") as f:
        f.write(f"# CTS swap probe — {CAM} extrinsics\n")
        f.write(f"# suv_eval scene_0220 frame 0 vs sedan_eval first-scene override\n\n")
        f.write(f"target raw extrinsic (suv_eval scene_0220 sample[0] {CAM}):\n")
        f.write(np.array2string(np.array(raw["extrinsics"][front_idx]),
                                precision=6, suppress_small=True) + "\n\n")
        f.write(f"sedan rig delta[{CAM}] (E_sedan(0) @ inv(E_target(0))):\n")
        f.write(np.array2string(sedan_map[CAM], precision=6, suppress_small=True) + "\n\n")
        f.write(f"reconstructed sedan extrinsic (delta @ raw target):\n")
        f.write(np.array2string(reconstructed, precision=6, suppress_small=True) + "\n\n")
        for cond in ["NORMAL", "EXT", "IMG", "CAL"]:
            f.write(f"{cond} extrinsic (loader output):\n")
            f.write(np.array2string(results[cond]["extrinsic"].numpy(),
                                    precision=6, suppress_small=True) + "\n\n")
    print(f"  wrote {OUT_TXT}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n[probe] === summary ===")
    n_pass = sum(1 for _, ok, _ in pass_fail if ok)
    n_total = len(pass_fail)
    for label, ok, _ in pass_fail:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    print(f"[probe] {n_pass}/{n_total} assertions passed")
    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
