# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""GOO-Real gaze-following dataset (cross-dataset evaluation only).

GOO-Real (`upeee/GOO-GAZE2021 <https://github.com/upeee/GOO-GAZE2021>`_) is a
retail-store gaze-following benchmark. This loader is used purely for
cross-dataset evaluation of a GazeFollow-trained :class:`VLMGazeModelGE`; it is
not wired for training.

It mirrors :meth:`omnigf.datasets.gazefollow.GazeFollowDataset.__getitem__`'s
output contract so a sample flows through
``UnifiedGazeFrameDataset -> VLMGazeDataset -> VLMGazeCollator`` unchanged.

Annotation format
-----------------
Each split is a pickled ``list[dict]``. Per sample:

* ``filename``    : Windows-style relative path, e.g. ``"73\\cam0\\...jpg"``.
  The real image lives at ``<root>/<img_subdir>/<filename with / separators>``
  and is stored at full resolution (1920x1080).
* ``hx, hy``      : gazer head centre, in a **640x480 annotation frame**.
* ``gaze_cx, gaze_cy`` : gaze-target point, same 640x480 frame.
* ``ann.bboxes`` / ``ann.labels`` : per-object boxes; **``label == 25`` is the
  gazer's head box** (exactly one per image, always contains ``(hx, hy)``).

Coordinates are normalised by the 640x480 annotation frame (``ANN_FRAME``) to
``[0, 1]``, then mapped to actual-image pixels for head cropping — identical to
the GazeFollow flow, which loads normalised boxes and scales by the image size.
GOO-Real gaze targets are always in-frame, so ``inout = 1`` for every sample
(in/out AP is not meaningful for this benchmark).
"""

import logging
import os
import pickle
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from omnigf.utils.utils import square_bbox
from omnigf.utils import generate_gaze_heatmap, generate_mask
from omnigf.train.transforms import Compose

logger = logging.getLogger(__name__)


# Head-class label in ``ann.labels`` for the gazer's head box.
GOO_HEAD_LABEL = 25

# Annotation coordinate frame (width, height). Head/gaze coordinates in the
# pickle live in this frame, not the actual image resolution. Confirmed by
# visual overlay.
ANN_FRAME_DEFAULT = (640, 480)

# Split -> annotation pickle filename.
GOO_SPLIT_FILES = {
    "test": "testrealhumansNew.pickle",
    "val": "valrealhumansNew.pickle",
}


class GOORealDataset(Dataset):
    """GOO-Real gaze-following dataset (single-person, eval-only)."""

    # Annotation coordinate frame (width, height); see ``ANN_FRAME_DEFAULT``.
    ANN_FRAME: Tuple[int, int] = ANN_FRAME_DEFAULT

    def __init__(
        self,
        root,
        ann_root,
        split: str = "test",
        transform: Union[Compose, None] = None,
        tr: tuple = (0.0, 0.0),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: Union[int, str] = "all",
        img_subdir: str = "finalrealdatasetImgsV3",
        ann_frame: Tuple[int, int] = None,
        **kwargs,
    ):
        super().__init__()
        assert split in GOO_SPLIT_FILES, (
            f"GOORealDataset supports splits {list(GOO_SPLIT_FILES)}, got {split!r}. "
            "GOO-Real is used for cross-dataset evaluation only."
        )
        self.root = root
        self.split = split
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.img_subdir = img_subdir
        if ann_frame is not None:
            self.ANN_FRAME = tuple(ann_frame)

        ann_path = os.path.join(ann_root, GOO_SPLIT_FILES[split])
        with open(ann_path, "rb") as f:
            self.data = pickle.load(f)
        logger.info(
            "Loaded %d GOO-Real %s samples from %s (ann_frame=%s)",
            len(self.data), split, ann_path, self.ANN_FRAME,
        )

    def __len__(self) -> int:
        return len(self.data)

    def _head_box_ann(self, entry: Dict) -> np.ndarray:
        """Return the gazer's head box ``[x1, y1, x2, y2]`` in the annotation
        frame: the ``label == 25`` box containing ``(hx, hy)`` (falls back to the
        first head box, then to a small box around ``(hx, hy)``)."""
        bboxes = entry["ann"]["bboxes"]
        labels = entry["ann"]["labels"]
        head_idx = np.where(labels == GOO_HEAD_LABEL)[0]
        hx, hy = entry["hx"], entry["hy"]
        if len(head_idx) > 0:
            for i in head_idx:
                x1, y1, x2, y2 = bboxes[i]
                if x1 <= hx <= x2 and y1 <= hy <= y2:
                    return bboxes[i].astype(np.float32)
            return bboxes[head_idx[0]].astype(np.float32)
        # Fallback: no head-labelled box — synthesise one around (hx, hy).
        fw, fh = self.ANN_FRAME
        half = 0.04 * fw
        return np.array([hx - half, hy - half, hx + half, hy + half], np.float32)

    def __getitem__(self, index: int) -> Dict:
        entry = self.data[index]

        # ---- Image ----
        rel = entry["filename"].replace("\\", "/")
        image = Image.open(
            os.path.join(self.root, self.img_subdir, rel)
        ).convert("RGB")
        img_w, img_h = image.size
        fw, fh = self.ANN_FRAME

        # ---- Head box + gaze point: annotation frame -> normalised [0, 1] ----
        hb_ann = self._head_box_ann(entry)
        head_bbox_norm = torch.tensor([
            hb_ann[0] / fw, hb_ann[1] / fh, hb_ann[2] / fw, hb_ann[3] / fh,
        ], dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(0)  # [1, 4]

        gaze_norm = torch.tensor([
            entry["gaze_cx"] / fw, entry["gaze_cy"] / fh,
        ], dtype=torch.float32).clamp(0.0, 1.0).unsqueeze(0)  # [1, 2]

        # ---- Head crop (in actual-image pixels), mirroring GazeFollow ----
        scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        head_bboxes_pix = square_bbox(head_bbox_norm * scale, img_w, img_h)
        heads = [image.crop(head_bboxes_pix[0].int().tolist())]

        # ---- Pad one dummy person at the front (annotated person is last) ----
        dummy_head = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        heads = [dummy_head] + heads
        head_bboxes_pix = torch.cat([torch.zeros((1, 4)), head_bboxes_pix])
        gaze_pts = torch.cat([torch.full((1, 2), -1.0), gaze_norm])
        inout = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        pids = torch.tensor([-1.0, 0.0], dtype=torch.float32)

        # ---- Normalise head boxes back to [0, 1] ----
        head_bboxes = head_bboxes_pix / scale

        n_people = len(heads)
        no_social = torch.zeros(n_people, dtype=torch.float32) - 1

        sample = {
            "image": image,
            "heads": heads,
            "head_bboxes": head_bboxes,
            "gaze_pts": gaze_pts,
            "inout": inout,
            "pids": pids,
            "coatt_labels": no_social.clone(),
            "lah_labels": no_social.clone(),
            "laeo_labels": no_social.clone(),
            "num_valid_people": torch.tensor(1),
            "is_child": no_social.clone(),
            "speaking": no_social.clone(),
            "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
            "path": [rel],
            "dataset": "goo_real",
        }

        # ---- Transform (resize + to-tensor + normalise) ----
        if self.transform:
            sample = self.transform(sample)

        # ---- Head masks / centres / gaze vectors ----
        _, t_h, t_w = sample["image"].shape
        sample["head_masks"] = generate_mask(sample["head_bboxes"], t_w, t_h)
        head_bboxes = sample["head_bboxes"]
        head_centers = torch.hstack([
            (head_bboxes[:, [0]] + head_bboxes[:, [2]]) / 2,
            (head_bboxes[:, [1]] + head_bboxes[:, [3]]) / 2,
        ])
        sample["head_centers"] = head_centers
        sample["gaze_vecs"] = F.normalize(
            sample["gaze_pts"] - head_centers, p=2, dim=-1,
        )

        # ---- Gaze heatmaps (all splits — GOO uses the single-annotator metric
        # path, unlike GazeFollow test). Dummy row (gaze = -1) stays all-zeros. ----
        sample["gaze_heatmaps"] = generate_gaze_heatmap(
            sample["gaze_pts"], sigma=self.heatmap_sigma, size=self.heatmap_size,
        )

        # ---- Add temporal dim (T=1) for compatibility, except pids/path/dataset ----
        for key, item in sample.items():
            if key not in ["path", "dataset", "pids"]:
                sample[key] = item.unsqueeze(0)

        return sample
