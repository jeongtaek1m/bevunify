"""Central augmentation manager for bevunify (keeps third_party untouched).

All augmentation flows through the shared dataloader, so whatever is enabled
here applies identically to all 6 models. Two augmentations live here:

  * image warp     -- resize / zoom / rotate / crop / flip, with the matching
                      intrinsic update baked in (so geometry stays consistent and
                      the per-model wrappers can keep ``post_rots = identity``).
                      Delegates to the host ``RandomTransformImage``; toggled by
                      ``data.augment_img`` (params in ``data.img_params``).

  * extrinsic noise -- a small random SE(3) perturbation applied per camera to the
                      *extrinsic that is fed to the model*. The image and the BEV
                      GT are left untouched, so the model is handed a deliberately
                      mis-calibrated camera pose -- a calibration-noise augmentation
                      that teaches robustness to extrinsic error. Toggled by
                      ``data.extrin_noise.enabled``.

Everything defaults OFF. Enable per experiment by selecting an ``augmentation``
preset (``config/augmentation/*.yaml``), e.g. on the CLI:
    python -m bevunify.train +experiment=cvt augmentation=warp_extrin
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

from GaussianLSS.data.augmentations import RandomTransformImage


class ImageWarp(RandomTransformImage):
    """bevunify image-warp augmentation.

    Identical to the host ``RandomTransformImage`` (resize/zoom/rotate/crop/flip +
    intrinsic adjust); subclassed so any future warp change is made here, in
    bevunify, rather than in the vendored GaussianLSS code.
    """
    pass


class ExtrinsicNoise:
    """Per-camera extrinsic (ego->cam) calibration noise.

    A random SE(3) ``delta`` is applied in the camera frame: ``E' = delta @ E``.
    ``delta`` is a small rotation (per-axis Gaussian, ``rot_std_deg``) plus a small
    translation (per-axis Gaussian, ``trans_std_m``). Independent noise per camera.

    The image and BEV GT are not changed, so the perturbed extrinsic is wrong
    relative to the true camera pose -- exactly the intended calibration-noise aug.
    No-op outside training and when both stds are 0.

    Args:
        rot_std_deg: std of per-axis rotation noise, in degrees.
        trans_std_m: std of per-axis translation noise, in meters.
        training:    only perturb during training.
    """

    def __init__(self, rot_std_deg=0.0, trans_std_m=0.0, training=True):
        self.rot_std_deg = float(rot_std_deg)
        self.trans_std_m = float(trans_std_m)
        self.training = training

    @property
    def active(self):
        return self.training and (self.rot_std_deg > 0.0 or self.trans_std_m > 0.0)

    def _sample_delta(self):
        delta = np.eye(4, dtype=np.float32)
        if self.rot_std_deg > 0.0:
            angles = np.random.normal(0.0, self.rot_std_deg, size=3)
            delta[:3, :3] = R.from_euler("xyz", angles, degrees=True).as_matrix()
        if self.trans_std_m > 0.0:
            delta[:3, 3] = np.random.normal(0.0, self.trans_std_m, size=3)
        return delta

    def __call__(self, extrinsics):
        """extrinsics: (N, 4, 4) ego->cam matrices. Returns a noised (N, 4, 4) copy."""
        extrinsics = np.asarray(extrinsics, dtype=np.float32).copy()
        if not self.active:
            return extrinsics
        for i in range(extrinsics.shape[0]):
            extrinsics[i] = self._sample_delta() @ extrinsics[i]
        return extrinsics


def build_extrinsic_noise(extrin_noise, training):
    """Build an ``ExtrinsicNoise`` from the ``data.extrin_noise`` config block, or
    ``None`` when disabled/absent. Accepts a dict / OmegaConf node with keys
    ``enabled``, ``rot_std_deg``, ``trans_std_m``."""
    if not extrin_noise:
        return None
    cfg = dict(extrin_noise)
    if not cfg.get("enabled", False):
        return None
    return ExtrinsicNoise(
        rot_std_deg=cfg.get("rot_std_deg", 0.0),
        trans_std_m=cfg.get("trans_std_m", 0.0),
        training=training,
    )
