"""GaussianLSS GT loader with per-model center/offset/visibility toggles.

This subclasses the host ``LoadDataTransform`` and re-implements only
``get_bev_from_gtbbox`` / ``__call__`` so that the (expensive) center-heatmap,
offset and visibility signals are produced *only* when the selected model's loss
consumes them. With all toggles on, the output is identical to the host loader.

Toggle flags arrive via ``cfg.data`` (see ``config/gt/*.yaml``):
    gt_center, gt_offset, gt_visibility  (all default True)
"""
import numpy as np
import cv2
import torch
from nuscenes.utils.data_classes import Box

from GaussianLSS.data.transforms import LoadDataTransform, Sample
from GaussianLSS.data.common import INTERPOLATION, sincos2quaternion

from .augmentation import ImageWarp, build_extrinsic_noise


class ToggleLoadDataTransform(LoadDataTransform):
    def __init__(self, *args, gt_center=True, gt_offset=True, gt_visibility=True,
                 extrin_noise=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gt_center = bool(gt_center)
        self.gt_offset = bool(gt_offset)
        self.gt_visibility = bool(gt_visibility)

        # Augmentation is managed centrally in bevunify.augmentation (third_party
        # left untouched). Image warp: re-wrap the host transform as ImageWarp so
        # future warp tweaks live in bevunify. Extrinsic noise: None unless enabled.
        training = kwargs.get("training", True)
        if self.augment_img is not None:
            self.augment_img = ImageWarp(kwargs.get("img_params"), training)
        self.extrin_noise = build_extrinsic_noise(extrin_noise, training)

    def get_bev_from_gtbbox(self, sample: Sample, bev_augm, mode="vehicle"):
        scene_dir = self.labels_dir / sample.scene
        gt_box = np.load(scene_dir / sample.gt_box, allow_pickle=True)["gt_box"]
        V = sample.view

        bev = np.zeros((200, 200), dtype=np.uint8)

        want_center_px = self.gt_center or self.gt_offset
        want_mask = self.gt_center or self.gt_offset or self.gt_visibility

        center_score = np.zeros((200, 200), dtype=np.float32) if self.gt_center else None
        center_offset = np.zeros((200, 200, 2), dtype=np.float32) if self.gt_offset else None
        visibility = np.full((200, 200), 255, dtype=np.uint8) if self.gt_visibility else None
        buf = np.zeros((200, 200), dtype=np.uint8) if want_mask else None
        coords = (
            np.stack(np.meshgrid(np.arange(200), np.arange(200)), -1).astype(np.float32)
            if want_center_px
            else None
        )
        sigma = 1

        for box_data in gt_box:
            if len(box_data) == 0:
                continue
            class_idx = int(box_data[7])
            if class_idx == 5 and mode == "vehicle":
                continue
            elif class_idx != 5 and mode == "ped":
                continue
            translation = [box_data[0], box_data[1], box_data[4]]
            size = [box_data[2], box_data[3], box_data[5]]
            yaw = box_data[6]
            yaw = -yaw - np.pi / 2
            visibility_token = box_data[8]
            box = Box(translation, size, sincos2quaternion(np.sin(yaw), np.cos(yaw)))
            points = box.bottom_corners()

            center = points.mean(-1)[:, None]

            homog_points = np.ones((4, 4))
            homog_points[:3, :] = points
            homog_points[-1, :] = 1
            points = self._prepare_augmented_boxes(bev_augm, homog_points)
            points[2] = 1
            points = (V @ points)[:2]
            cv2.fillPoly(bev, [points.round().astype(np.int32).T], 1, INTERPOLATION)

            if want_center_px:
                homog_points = np.ones((4, 1))
                homog_points[:3, :] = center
                homog_points[-1, :] = 1
                center = self._prepare_augmented_boxes(bev_augm, homog_points).astype(np.float32)
                center[2] = 1
                center = (V @ center)[:2, 0].astype(np.float32)

            if want_mask:
                buf.fill(0)
                cv2.fillPoly(buf, [points.round().astype(np.int32).T], 1, INTERPOLATION)
                mask = buf > 0

            if want_center_px:
                center_off = center[None] - coords
                if self.gt_offset:
                    center_offset[mask] = center_off[mask]
                if self.gt_center:
                    g = np.exp(-(center_off ** 2).sum(-1) / (2 * sigma ** 2))
                    center_score = np.maximum(center_score, g)

            if self.gt_visibility:
                visibility[mask] = visibility_token

        bev = self.to_tensor(255 * bev)
        center_score = self.to_tensor(center_score) if self.gt_center else None
        center_offset = self.to_tensor(center_offset) if self.gt_offset else None
        visibility = torch.from_numpy(visibility) if self.gt_visibility else None

        return bev, center_score, center_offset, visibility

    def _assign_bev(self, result, key, bev, center, offset, visibility):
        result[key] = bev
        if center is not None:
            result[f"{key}_center"] = center
        if offset is not None:
            result[f"{key}_offset"] = offset
        if visibility is not None:
            result[f"{key}_visibility"] = visibility

    def __call__(self, batch):
        if not isinstance(batch, Sample):
            batch = Sample(**batch)

        # The generated JSON has no 'cam_ids'; the host only exercises the augment
        # path (get_cameras_augm) which doesn't use them. The non-augment path
        # (get_cameras) and CVT's cam_idx need them — the dataset was generated with
        # all 6 cameras, so they are simply range(len(images)).
        if "cam_ids" not in batch:
            batch.cam_ids = list(range(len(batch.images)))

        # Extrinsic calibration-noise aug: perturb the per-camera extrinsic fed to
        # the model; the image and BEV GT stay clean. Recompute the cached cam->ego
        # so wrappers reading 'ego_from_cam' see the noise too. Train-only (active).
        if self.extrin_noise is not None and self.extrin_noise.active:
            noised = self.extrin_noise(batch.extrinsics)
            batch.extrinsics = noised
            if "ego_from_cam" in batch:
                batch.ego_from_cam = np.linalg.inv(noised)

        result = dict()
        result["view"] = torch.tensor(batch.view)
        result["token"] = batch["token"]
        result["map_name"] = batch["map_name"]
        result["pose"] = np.float32(batch["pose"])
        result["pose_inverse"] = np.float32(batch["pose_inverse"])

        if self.image_data:
            get_cameras = self.get_cameras_augm if self.augment_img is not None else self.get_cameras
            result.update(get_cameras(batch, **self.image_config))
            # The augment path (get_cameras_augm) leaves cam_idx commented out;
            # the non-augment path sets it. CVT (and any cam_idx consumer) needs it,
            # so re-inject when the image-warp path dropped it.
            if "cam_idx" not in result:
                result["cam_idx"] = torch.LongTensor(batch.cam_ids)
            # materialized cam->ego convention from bevunify.datagen (if present).
            # (Valid as-is while augment_bev is off; with BEV aug it would need
            #  pre-multiplying by inv(bev_augm) to stay consistent with extrinsics.)
            if "ego_from_cam" in batch:
                result["ego_from_cam"] = torch.tensor(np.float32(batch["ego_from_cam"]))

        if self.bev:
            bev_augm = self.augment_bev() if self.augment_bev else np.eye(4)

            if self.vehicle:
                self._assign_bev(result, "vehicle", *self.get_bev_from_gtbbox(batch, bev_augm, mode="vehicle"))
            if self.ped:
                self._assign_bev(result, "ped", *self.get_bev_from_gtbbox(batch, bev_augm, mode="ped"))

            if len(self.map_layers) > 0:
                result.update(self.get_map(batch, bev_augm))

            # Apply the BEV augmentation to whichever camera parameterisation is present.
            # (get_cameras returns only 'lidar2img' or only 'intrinsics'+'extrinsics'
            #  depending on split_intrin_extrin; get_cameras_augm returns both.)
            bev_augm = torch.from_numpy(bev_augm).float()
            if "extrinsics" in result:
                result["extrinsics"] = result["extrinsics"] @ bev_augm
            if "lidar2img" in result:
                result["lidar2img"] = result["lidar2img"] @ bev_augm
            result["bev_augm"] = bev_augm

        return result
