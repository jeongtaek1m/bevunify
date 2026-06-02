"""cross_view_transformers wrapper.

CVT's forward already takes the batch dict {image, intrinsics, extrinsics, cam_idx}
and returns a dict whose keys are exactly CVT's ``outputs`` keys. By setting those
keys to the canonical names (``vehicle`` / ``vehicle_center``), CVT's output already
matches the GaussianLSS loss/metric contract — no output remapping needed.

Needs the unified DataModule run with split_intrin_extrin=True so the batch carries
``intrinsics``/``extrinsics``/``cam_idx``.
"""
import torch.nn as nn
from omegaconf import OmegaConf
from hydra.utils import instantiate

from .geom import add_repo_to_path
from .repo_compose import compose_repo_cfg


class CVTWrapper(nn.Module):
    def __init__(self, key, repo_root, config_name="config", center=True,
                 experiment="cvt_nuscenes_vehicle", backbone="efficientnet-b4",
                 image_h=None, image_w=None):
        super().__init__()
        self.key = key
        repo_root = add_repo_to_path(repo_root)

        cfg = compose_repo_cfg(
            config_dir=f"{repo_root}/config",
            config_name=config_name,
            overrides=[f"+experiment={experiment}"],
        )
        # Build outputs with the canonical keys (NOTE: ${key} can't interpolate into a
        # YAML dict KEY, so we set them here) -> CVT emits {vehicle, vehicle_center}.
        outputs = {key: [0, 1]}
        if center:
            outputs[f"{key}_center"] = [1, 2]
        OmegaConf.set_struct(cfg, False)
        # image resolution: CVT sizes its EfficientNet output + cross-view positional
        # embeddings to data.image (the CVT repo's own default 224x480). Push the unified
        # loader's resolution so the embedding grid matches the actual input.
        if image_h is not None and OmegaConf.select(cfg, "data.image") is not None:
            cfg.data.image.h = image_h
            cfg.data.image.w = image_w
        # backbone knob (CVT EfficientNetExtractor; native efficientnet-b4 / -b0).
        cfg.model.encoder.backbone.model_name = backbone
        cfg.model.outputs = OmegaConf.create(outputs)
        self.net = instantiate(cfg.model)

    def forward(self, batch):
        # CVT consumes the dict directly; output keys == canonical keys.
        # Extrinsic convention: ego→cam (cam_from_ego). Matches GaussianLSS loader
        # exactly — verified against CVT's native nuscenes_dataset.py:174 and
        # encoder.py:324 (E_inv = batch['extrinsics'].inverse()). VR swap propagates
        # via single inversion inside the encoder; see tests/probe_per_model_extrinsic_forward.py.
        return self.net(batch)
