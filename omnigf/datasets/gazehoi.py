# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""GazeHOI training/eval dataset for the VLM pipeline.

Reads ``gazehoi_{split}.h5`` (one row per image, per-row ``[K, ...]`` arrays
for the K (head, target_object) pairs in that image — produced by
``scripts/llm_label_mapping/build_gazehoi_h5.py``) and emits **one
(head, target) pair per ``__getitem__``**, mirroring the original
``semgaze/semgaze/datasets/gazehoi.py:__getitem__`` semantic. Multi-target
heads contribute multiple samples (one per target); every annotation
supervises / evaluates independently.

The output schema matches :class:`GazeFollowDataset` so the
:class:`UnifiedGazeFrameDataModule` can mix this source into the standard
VLM pipeline without further changes — the leading ``unsqueeze(0)`` at the
end is the temporal-compat dim the rest of the codebase expects.

Bad-pair filter
---------------
~2% of pairs in the train/val/test h5 have ``head_bboxes`` /
``gaze_points`` / ``object_bboxes`` outside the normalised ``[0, 1]``
range. Cause is a per-image upstream-packaging inconsistency: a small
subset of GazeHOI images appear to have been rotated and/or resized
(possibly down-sampled from much larger originals) when packaged, but
their CSV annotations were left in the original-image pixel coordinates.
``build_gazehoi_h5.py`` then normalised by the on-disk PIL size, so the
resulting values overflow ``[0, 1]`` for those rows. Recovery would
require knowing the original dims AND any orientation transform per
image, neither of which is available from the dataset. We drop these
pairs at ``__init__`` time (with a single aggregated warning, not
silently). ``len(dataset)`` reflects the post-filter pair count.

Most images in the dataset are fine — the on-disk sizes are
heterogeneous (e.g. ``640×480``, ``1920×1080``, ``683×512``,
``512×512``, ...), and rows where the CSV pixel coords fit inside the
corresponding on-disk PIL size normalise correctly. Only the
problematic subset is filtered.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

from omnigf.utils.utils import square_bbox
from omnigf.utils import (
    generate_gaze_heatmap,
    generate_mask,
)
from omnigf.train.transforms import (
    Compose,
    RandomHeadBboxJitter,
)


class GazeHOIDataset(Dataset):
    """Per-pair flattening of ``gazehoi_{split}.h5`` for VLM training.

    Each ``__getitem__`` returns one ``(image, head, target_object)`` pair
    as a single-person sample (``K = 1``). Pixels are squared and cropped
    in pixel space (matching ``gazefollow.py:454-477``); bboxes are
    re-normalised + clamped to ``[0, 1]`` for the VLM prompt. The Compose
    transform from the unified pipeline runs Resize + ToTensor (no
    Normalize, since ``UnifiedGazeFrameDataModule`` defaults
    ``skip_normalize=True``); a leading temporal singleton dim is appended
    at the end.
    """

    def __init__(
        self,
        root: str,
        ann_root: str,
        split: str = "train",
        transform: Optional[Compose] = None,
        tr: Tuple[float, float] = (0.0, 0.0),
        heatmap_size: int = 64,
        heatmap_sigma: int = 3,
        vocab2id_path: Optional[str] = None,
        use_semantic_labels: bool = True,
        return_head_mask: bool = False,
        # Accepted-and-ignored kwargs for unified-pipeline compat
        # (e.g. ``num_people``, ``head_thr`` — GazeHOI has no per-image
        # multi-person logic so these don't apply).
        **_unused,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), (
            f"Expected `split` to be one of ['train', 'val', 'test'], got {split!r}"
        )

        self.root = root
        self.ann_root = ann_root
        self.split = split
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.use_semantic_labels = use_semantic_labels
        self.return_head_mask = return_head_mask
        # Train-time random head jitter; identity at val/test (tr=(0,0)).
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)

        # Load h5 (one row per image; per-row [K, ...] arrays).
        h5_path = os.path.join(ann_root, f"gazehoi_{split}.h5")
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"GazeHOI annotations not found: {h5_path}")
        self.df = pd.read_hdf(h5_path, "data")

        # Flat (row_idx, pair_idx_within_row) index — __len__ becomes the
        # total pair count, with bad pairs filtered out (see module
        # docstring for the cause).
        self.index_map: List[Tuple[int, int]] = []
        n_total_pairs = 0
        n_bad_pairs = 0
        for row_idx in range(len(self.df)):
            row = self.df.iloc[row_idx]
            head_bboxes = np.asarray(row["head_bboxes"], dtype=np.float32)
            gaze_points = np.asarray(row["gaze_points"], dtype=np.float32)
            object_bboxes = np.asarray(row["object_bboxes"], dtype=np.float32)
            n_pairs = int(head_bboxes.shape[0])
            for pair_idx in range(n_pairs):
                n_total_pairs += 1
                hb = head_bboxes[pair_idx]
                gp = gaze_points[pair_idx]
                ob = object_bboxes[pair_idx] if pair_idx < object_bboxes.shape[0] else None
                # Reject pairs whose normalised coordinates fall outside
                # [0, 1] in any field — the source CSV's pixel coords
                # referenced the original (pre-resize) image dimensions
                # while the on-disk images have been downsized, so the
                # normalisation in ``build_gazehoi_h5.py`` overflowed.
                bad = bool(
                    (hb > 1.0).any() or (hb < 0.0).any()
                    or (gp > 1.0).any() or (gp < 0.0).any()
                )
                if ob is not None:
                    bad = bad or bool(
                        (ob > 1.0).any() or (ob < 0.0).any()
                    )
                if bad:
                    n_bad_pairs += 1
                    continue
                self.index_map.append((row_idx, pair_idx))
        if n_bad_pairs > 0:
            logger.warning(
                "GazeHOIDataset[%s]: dropped %d/%d pairs (%.2f%%) whose "
                "normalised coords overflow [0, 1]; for those rows the "
                "CSV pixel coords exceed the on-disk PIL image size — see "
                "module docstring.",
                split, n_bad_pairs, n_total_pairs,
                100.0 * n_bad_pairs / max(n_total_pairs, 1),
            )

        # Vocab2id for SemGaze-style val acc@1/@3. OOV labels resolve to -1
        # (the metric module's ``ignore_index``).
        self.vocab2id: Dict[str, int] = {}
        if vocab2id_path is not None and os.path.isfile(vocab2id_path):
            with open(vocab2id_path) as f:
                self.vocab2id = json.load(f)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Dict:
        row_idx, pair_idx = self.index_map[index]
        row = self.df.iloc[row_idx]
        path = row["path"]

        # Load image.
        img_path = os.path.join(self.root, path)
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Per-pair fields, lifted out of the row's [K, ...] arrays.
        head_bbox_norm = np.asarray(row["head_bboxes"], dtype=np.float32)[pair_idx]
        gaze_pt_norm = np.asarray(row["gaze_points"], dtype=np.float32)[pair_idx]
        pair_id = int(np.asarray(row["person_ids"], dtype=np.int64)[pair_idx])
        # Object bbox for the same pair (the looked-at object). Only
        # used by the visualisation script; downstream losses don't
        # consume this field. ``[xmin, ymin, xmax, ymax]`` in normalised
        # coords; bad-pair filter at ``__init__`` already rejected
        # rows whose ``object_bboxes`` overflowed [0, 1].
        obj_bbox_norm = np.asarray(
            row["object_bboxes"], dtype=np.float32,
        )[pair_idx]
        label_text = ""
        if self.use_semantic_labels:
            sem_arr = np.asarray(row["semgaze_head_labels"], dtype=object)
            if pair_idx < sem_arr.shape[0]:
                label_text = str(sem_arr[pair_idx])

        # Pixel-space → jitter (train) → square → crop head from PIL.
        # Matches gazefollow.py:454-464 + videoattentiontarget_temporal.py:368-376.
        scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float)
        head_bbox_px = torch.from_numpy(head_bbox_norm).float().unsqueeze(0) * scale
        if self.split == "train":
            head_bbox_px = self.jitter_bbox(head_bbox_px, img_w, img_h)
        head_bbox_px = square_bbox(head_bbox_px, img_w, img_h)
        head_crop = image.crop(head_bbox_px[0].int().tolist())

        # Re-normalise + clamp the squared bbox for the VLM prompt.
        head_bboxes = (head_bbox_px / scale).clamp(0.0, 1.0)  # [1, 4]
        gaze_pts = torch.from_numpy(gaze_pt_norm).float().unsqueeze(0)  # [1, 2]
        object_bboxes = torch.from_numpy(obj_bbox_norm).float().unsqueeze(0)  # [1, 4]
        inout = torch.ones(1, dtype=torch.float)
        pids = torch.tensor([pair_id], dtype=torch.long)

        # Empty social labels — K=1 so P=K*(K-1)=0 pair slots.
        empty_pair = torch.zeros(0, dtype=torch.float) - 1

        sample: Dict = {
            "image": image,
            "heads": [head_crop],
            "head_bboxes": head_bboxes,
            "object_bboxes": object_bboxes,
            "gaze_pts": gaze_pts,
            "inout": inout,
            "pids": pids,
            "coatt_labels": empty_pair.clone(),
            "lah_labels": empty_pair.clone(),
            "laeo_labels": empty_pair.clone(),
            "num_valid_people": torch.tensor(1),
            "is_child": torch.full((1,), -1, dtype=torch.float),
            "speaking": torch.full((1,), -1, dtype=torch.float),
            "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
            "path": [path],
            "dataset": "gazehoi",
        }

        # Semantic-label fields (text + ids/embs). InfoNCE-style fields are
        # populated with placeholders so the unified pipeline's optional
        # plumbing doesn't choke on missing keys when other components
        # (e.g. ``_eval_text_semantic``) check for them.
        if self.use_semantic_labels:
            sample["gaze_label_texts"] = [label_text]
            tid = int(self.vocab2id.get(label_text, -1)) if label_text else -1
            sample["gaze_test_label_ids"] = torch.tensor([tid], dtype=torch.long)
            sample["gaze_label_ids"] = torch.tensor([tid], dtype=torch.long)
            sample["gaze_label_embs"] = torch.zeros(1, 512)
            sample["gaze_multi_test_label_ids"] = torch.full((1, 5), -1, dtype=torch.long)

        # Apply unified Compose ([Resize, ToTensor]) — turns ``image`` into
        # [3, H, W] tensor and ``heads`` from PIL list into [1, 3, hh, hw].
        if self.transform is not None:
            sample = self.transform(sample)

        # Head centers for downstream gaze-vec computation.
        hb = sample["head_bboxes"]
        head_centers = torch.hstack([
            (hb[:, [0]] + hb[:, [2]]) / 2,
            (hb[:, [1]] + hb[:, [3]]) / 2,
        ])
        sample["head_centers"] = head_centers

        # 2D gaze vector (post-resize, per-person).
        sample["gaze_vecs"] = F.normalize(
            sample["gaze_pts"] - head_centers, p=2, dim=-1,
        )

        # Heatmap supervision target.
        sample["gaze_heatmaps"] = generate_gaze_heatmap(
            sample["gaze_pts"], sigma=self.heatmap_sigma, size=self.heatmap_size,
        )

        # Optional head mask.
        if self.return_head_mask and isinstance(sample["image"], torch.Tensor):
            _, sh, sw = sample["image"].shape
            sample["head_masks"] = generate_mask(sample["head_bboxes"], sw, sh)

        # Temporal-compat: every per-frame tensor gets a leading singleton
        # time dim, matching ``GazeFollowDataset.__getitem__:590-593``. The
        # listed exceptions are the same: ``path`` and ``dataset`` are
        # Python objects, ``pids`` is treated as a global identifier
        # (no time dim), and ``gaze_label_texts`` is a list of str.
        for key, item in sample.items():
            if key in ("path", "dataset", "pids", "gaze_label_texts"):
                continue
            if isinstance(item, torch.Tensor):
                sample[key] = item.unsqueeze(0)

        return sample
