# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import logging
import os
from typing import Dict, Union, Tuple

logger = logging.getLogger(__name__)

from omegaconf import OmegaConf, ListConfig

import numpy as np
import pandas as pd
from PIL import Image

import torch
import json
import lightning.pytorch as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from omnigf.utils.utils import square_bbox
from omnigf.utils.image import IMG_MEAN, IMG_STD
from omnigf.utils.social_gaze import get_lah_labels, get_shuffle_idx
from omnigf.utils import (
    generate_gaze_heatmap,
    generate_mask,
    pair,
)
from omnigf.train.transforms import (
    RandomHeadBboxJitter,
    RandomHorizontalFlip,
    RandomCropSafeGaze,
    ColorJitter,
    Normalize,
    ToTensor,
    Compose,
    Resize,
)


# ============================================================================= #
#                               GAZEFOLLOW DATASET                              #
# ============================================================================= #
class GazeFollowDataset(Dataset):
    def __init__(
        self,
        root,
        ann_root,
        split: str = "train",
        transform: Union[Compose, None] = None,
        tr: tuple = (-0.1, 0.1),
        heatmap_sigma: int = 3,
        heatmap_size: int = 64,
        num_people: int = 5,
        head_thr: float = 0.5,
        return_head_mask: bool = False,
        include_inout: bool = False,
        use_semgaze_bbox: bool = False,
        semgaze_ann_root: str = None,
        use_semantic_labels: bool = False,
        semantic_vocab_dir: str = None,
        prediction_json: str = None,
    ):
        super().__init__()

        assert split in ("train", "val", "test"), (
            f"Expected `split` to be one of [`train`, `val`, `test`] but received `{split}` instead."
        )
        assert (num_people == "all") or (num_people > 0), (
            f'Expected `num_people` to be strictly positive or "all", but received {num_people} instead.'
        )
        assert 0 <= head_thr <= 1, (
            f"Expected `head_thr` to be in [0, 1]. Received {head_thr} instead."
        )

        self.root = root
        self.ann_root = ann_root
        self.split = split
        self.jitter_bbox = RandomHeadBboxJitter(p=1.0, tr=tr)
        self.transform = transform
        self.heatmap_sigma = heatmap_sigma
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.head_thr = head_thr
        self.return_head_mask = return_head_mask
        self.include_inout = include_inout
        self.use_semgaze_bbox = use_semgaze_bbox
        self.use_semantic_labels = use_semantic_labels

        # Load semantic vocabulary embeddings
        self._train_label2id = None
        self._train_vocab_emb = None
        self._test_label2id = None
        self._test_vocab_emb = None
        if self.use_semantic_labels and semantic_vocab_dir is not None:
            train_emb_path = os.path.join(semantic_vocab_dir, "train_vocab_embeddings.pt")
            train_lab_path = os.path.join(semantic_vocab_dir, "train_label2id.json")
            if os.path.exists(train_emb_path):
                self._train_vocab_emb = torch.load(train_emb_path, map_location="cpu", weights_only=False)
                self._train_label2id = json.load(open(train_lab_path))
            test_emb_path = os.path.join(semantic_vocab_dir, "test_vocab_embeddings.pt")
            test_lab_path = os.path.join(semantic_vocab_dir, "test_vocab_labels.json")
            if os.path.exists(test_emb_path):
                self._test_vocab_emb = torch.load(test_emb_path, map_location="cpu", weights_only=False)
                test_labels = json.load(open(test_lab_path))
                self._test_label2id = {l: i for i, l in enumerate(test_labels)}

        # load annotations — use semgaze-merged h5 if semantic labels or semgaze bbox requested
        use_semgaze_h5 = self.use_semgaze_bbox or self.use_semantic_labels
        if use_semgaze_h5:
            semgaze_root = semgaze_ann_root or ann_root
            if self.split == "test":
                annt_path = os.path.join(semgaze_root, f"gazefollow_{self.split}_with_semgaze.h5")
            else:
                annt_path = os.path.join(semgaze_root, f"gazefollow_{self.split}_with_semgaze_corrected.h5")
            print(f"Loading semgaze-merged gazefollow annotations from {annt_path}")
        elif self.split == "train" and self.include_inout:
            print(f"Loading extended gazefollow train annotations from {ann_root}")
            annt_path = os.path.join(ann_root, f"gazefollow_train_extended.h5")
        else:
            annt_path = os.path.join(ann_root, f"gazefollow_{self.split}.h5")

        self.annotations = pd.read_hdf(
            annt_path, "data"
        )
        self.annotations = self.annotations.groupby("path")
        self.paths = list(self.annotations.groups.keys())
        self.length = len(self.paths)

        # Optional: load per-image gaze predictions and override the GT
        # gaze coordinates in __getitem__ (before shuffle/jitter/transform).
        # Format expected: a JSON list of records, each with "image_path"
        # and "gaze_pts_pred" (list of [x, y] in normalised [0,1] coords).
        # Saved by ``scripts/save_vlm_predictions.py`` with
        # ``+inference.split=train``.  Predictions cover the LAST K_save
        # h5 persons (since save runs with shuffle off and num_keep selects
        # from the end), so we map them to h5 positions [N-K_save, N-1].
        # For h5 positions outside that range, the GT is kept.
        self._predictions_by_path = None
        if prediction_json is not None:
            import json as _json
            print(f"GazeFollow: loading prediction overrides from {prediction_json}")
            with open(prediction_json) as f:
                _records = _json.load(f)
            self._predictions_by_path = {}
            for r in _records:
                p = r.get("image_path")
                if p is None:
                    continue
                self._predictions_by_path[p] = r.get("gaze_pts_pred", [])
            print(
                f"GazeFollow: loaded {len(self._predictions_by_path)} prediction "
                f"records (will override GT gaze in __getitem__)."
            )

    def __getitem__(self, index: int) -> Dict:
        # load annotations
        path = self.paths[index]
        img_annotations = self.annotations.get_group(path).iloc[0]
        pids = torch.from_numpy(img_annotations["person_ids"])
        inout = torch.from_numpy(img_annotations["inout"].astype(np.float32))

        if self.split != "test":
            gaze_pts = torch.from_numpy(
                img_annotations["gaze_points"].astype(np.float32)
            )
        else:
            gaze_pts = torch.from_numpy(
                img_annotations["gaze_points_p1"].astype(np.float32)
            )

        # Optional override: replace GT gaze with saved predictions.
        # Saved gaze_pts_pred[k] corresponds to h5 position
        # (N - K_save + k), since save was done with shuffle off and
        # num_keep selects the last K. We do this BEFORE shuffle / jitter
        # / transform so the predicted coords undergo the same augmentation
        # as the image and stay aligned.
        if self._predictions_by_path is not None and path in self._predictions_by_path:
            pred_list = self._predictions_by_path[path]
            N = gaze_pts.shape[0]
            K = len(pred_list)
            offset = N - K
            if offset >= 0:
                for k in range(K):
                    gp = pred_list[k]
                    if gp is None or gp[0] is None:
                        continue
                    gaze_pts[offset + k, 0] = float(gp[0])
                    gaze_pts[offset + k, 1] = float(gp[1])

        # Load image
        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        img_w, img_h = image.size

        # Load head bboxes (optionally replace with semgaze bboxes for matched targets)
        head_bboxes = img_annotations["head_bboxes"].astype(np.float32)
        if self.use_semgaze_bbox:
            sem_bboxes = img_annotations["semgaze_head_bboxes"].astype(np.float32)
            is_target = img_annotations["semgaze_head_is_target"].astype(np.int8)
            mask = (is_target == 1) & (sem_bboxes[:, 0] >= 0)
            head_bboxes[mask] = sem_bboxes[mask]
        head_bboxes = torch.from_numpy(head_bboxes)
        head_bboxes = head_bboxes * torch.tensor([img_w, img_h, img_w, img_h])
        # # expand bbox slightly
        # head_bboxes = expand_bbox(head_bboxes, img_w, img_h, k=0.1)

        # Load semantic labels (per-head)
        sem_labels = None  # list of str
        sem_embs = None    # [N, 512]
        sem_label_ids = None  # [N] train vocab IDs
        sem_test_label_ids = None  # [N] test vocab IDs
        sem_multi_test_label_ids = None  # [N, 5] multi-label test vocab IDs
        if self.use_semantic_labels and "semgaze_head_labels" in img_annotations.index:
            raw_labels = img_annotations["semgaze_head_labels"]  # ndarray of str
            sem_labels = list(raw_labels)
            N = len(sem_labels)
            # Build per-person CLIP embeddings and vocab IDs
            sem_embs = torch.zeros(N, 512)
            sem_label_ids = torch.full((N,), -1, dtype=torch.long)
            sem_test_label_ids = torch.full((N,), -1, dtype=torch.long)
            for i, label in enumerate(sem_labels):
                if label and label != "":
                    # Train vocab: embedding + ID (used for InfoNCE loss)
                    if label in self._train_label2id:
                        tid = self._train_label2id[label]
                        sem_label_ids[i] = tid
                        sem_embs[i] = self._train_vocab_emb[tid]
                    else:
                        # OOV labels are expected after an LLM-correction merge
                        # (the LLM emits free-text not present in the 3,790-label
                        # train vocab). Warn once per unique label per process to
                        # keep the signal without drowning the progress bar or
                        # slowing I/O across 100k+ samples.
                        if not hasattr(self, "_warned_oov_train_labels"):
                            self._warned_oov_train_labels = set()
                        if label not in self._warned_oov_train_labels:
                            self._warned_oov_train_labels.add(label)
                            logger.debug(
                                "Semantic label '%s' not in train vocab (first seen at path=%s)",
                                label, path,
                            )
                    # Test vocab: ID only (used for acc@1/acc@3 metrics)
                    if label in self._test_label2id:
                        sem_test_label_ids[i] = self._test_label2id[label]
                    elif self.split == "test":
                        if not hasattr(self, "_warned_oov_test_labels"):
                            self._warned_oov_test_labels = set()
                        if label not in self._warned_oov_test_labels:
                            self._warned_oov_test_labels.add(label)
                            logger.debug(
                                "Semantic label '%s' not in test vocab (first seen at path=%s)",
                                label, path,
                            )

            # Multi-label test IDs (test split only, from merged h5)
            if "semgaze_head_multi_test_label_ids" in img_annotations.index:
                raw_multi = img_annotations["semgaze_head_multi_test_label_ids"]  # [N, 5]
                sem_multi_test_label_ids = torch.from_numpy(
                    raw_multi.astype(np.int64)
                )  # [N, 5]

        # Shuffle people
        if self.split == "train" and len(head_bboxes) > 2:
            rand_indices = get_shuffle_idx(inout.numpy())
            pids = pids[rand_indices]
            head_bboxes = head_bboxes[rand_indices]
            gaze_pts = gaze_pts[rand_indices]
            inout = inout[rand_indices]
            if sem_labels is not None:
                sem_labels = [sem_labels[i] for i in rand_indices]
                sem_embs = sem_embs[rand_indices]
                sem_label_ids = sem_label_ids[rand_indices]
                sem_test_label_ids = sem_test_label_ids[rand_indices]
                if sem_multi_test_label_ids is not None:
                    sem_multi_test_label_ids = sem_multi_test_label_ids[rand_indices]

        # Jitter head bboxes
        if self.split == "train":
            head_bboxes = self.jitter_bbox(head_bboxes, img_w, img_h)

        # Square head bboxes (can have negative values)
        head_bboxes = square_bbox(head_bboxes, img_w, img_h)

        # Extract Heads (negative values add padding)
        heads = []
        for head_bbox in head_bboxes:
            heads.append(image.crop(head_bbox.int().tolist()))  # type:ignore

        # Select {1, num_people} people
        num_heads = len(heads)
        num_keep = num_heads
        if self.num_people != "all":
            if num_heads > 1:
                num_keep = np.random.randint(
                    2, min(num_heads, self.num_people) + 1
                )  # min 2 people
        #                 num_keep = np.random.randint(1, min(num_heads, self.num_people)+1)   # min one person
        head_bboxes = head_bboxes[-num_keep:]
        heads = heads[-num_keep:]
        pids = pids[-num_keep:]
        if self.split != "test":
            gaze_pts = gaze_pts[-num_keep:]
            inout = inout[-num_keep:]
        if sem_labels is not None:
            sem_labels = sem_labels[-num_keep:]
            sem_embs = sem_embs[-num_keep:]
            sem_label_ids = sem_label_ids[-num_keep:]
            sem_test_label_ids = sem_test_label_ids[-num_keep:]
            if sem_multi_test_label_ids is not None:
                sem_multi_test_label_ids = sem_multi_test_label_ids[-num_keep:]
        num_heads = len(heads)
        # pad at least one person
        num_missing_heads = (
            max(self.num_people + 1 - num_heads, 1) if self.num_people != "all" else 1
        )

        # Pad missing people (ie. heads, head_bboxes, gaze_pt and coatt); always have one extra person
        if num_missing_heads > 0:
            head_bboxes = torch.cat([torch.zeros((num_missing_heads, 4)), head_bboxes])
            heads = (
                num_missing_heads
                * [Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))]
                + heads
            )
            gaze_pts = torch.cat([torch.zeros((num_missing_heads, 2)) - 1, gaze_pts])
            pids = torch.cat([torch.zeros(num_missing_heads) - 1, pids])
        if self.split != "test":
            inout = torch.cat(
                [torch.zeros((num_missing_heads,), dtype=torch.float32) - 1, inout]
            )
        if sem_labels is not None and num_missing_heads > 0:
            sem_labels = [""] * num_missing_heads + sem_labels
            sem_embs = torch.cat([torch.zeros(num_missing_heads, 512), sem_embs])
            sem_label_ids = torch.cat([
                torch.full((num_missing_heads,), -1, dtype=torch.long), sem_label_ids,
            ])
            sem_test_label_ids = torch.cat([
                torch.full((num_missing_heads,), -1, dtype=torch.long), sem_test_label_ids,
            ])
            if sem_multi_test_label_ids is not None:
                K = sem_multi_test_label_ids.shape[1]  # MAX_MULTI_LABELS
                sem_multi_test_label_ids = torch.cat([
                    torch.full((num_missing_heads, K), -1, dtype=torch.long),
                    sem_multi_test_label_ids,
                ])

        # Normalize Head Bboxes
        head_bboxes /= torch.tensor([img_w, img_h, img_w, img_h], dtype=float)

        # Get LAH labels
        pairs_all = img_annotations["pairs"]
        lah_labels_all = img_annotations["lah_pairs"]
        lah_labels = get_lah_labels(pids, pairs_all, lah_labels_all)

        # Build Sample
        sample = {
            "image": image,
            "heads": heads,
            "head_bboxes": head_bboxes,
            "gaze_pts": gaze_pts,
            "inout": inout,
            "pids": pids,
            "coatt_labels": torch.zeros_like(lah_labels) - 1,
            "lah_labels": lah_labels,
            "laeo_labels": torch.zeros_like(lah_labels) - 1,
            "num_valid_people": torch.tensor(num_heads),
            "is_child": torch.zeros(len(heads), dtype=torch.float) - 1,
            "speaking": torch.zeros(len(heads), dtype=torch.float) - 1,
            "img_size": torch.tensor((img_w, img_h), dtype=torch.long),
            "path": [path],
            "dataset": "gazefollow",
        }

        if sem_labels is not None:
            sample["gaze_label_embs"] = sem_embs           # [N, 512]
            sample["gaze_label_ids"] = sem_label_ids        # [N] train vocab IDs
            sample["gaze_test_label_ids"] = sem_test_label_ids  # [N] test vocab IDs
            sample["gaze_label_texts"] = sem_labels         # list of str, len N
            if sem_multi_test_label_ids is not None:
                sample["gaze_multi_test_label_ids"] = sem_multi_test_label_ids  # [N, 5]

        # Transform
        if self.transform:
            sample = self.transform(sample)

        # compute head masks
        _, img_h, img_w = sample["image"].shape
        sample["head_masks"] = generate_mask(sample["head_bboxes"], img_w, img_h)

        # Compute Head Bbox Centers
        head_bboxes = sample["head_bboxes"]
        gaze_pts = sample["gaze_pts"]
        head_centers = torch.hstack(
            [
                (head_bboxes[:, [0]] + head_bboxes[:, [2]]) / 2,
                (head_bboxes[:, [1]] + head_bboxes[:, [3]]) / 2,
            ]
        )
        sample["head_centers"] = head_centers

        # compute 2d gaze vectors
        if self.split == "test":
            # only consider annotated person
            head_centers = head_centers[-1].unsqueeze(0)
        sample["gaze_vecs"] = F.normalize(gaze_pts - head_centers, p=2, dim=-1)

        if self.split != "test":
            # generate gaze heatmaps
            sample["gaze_heatmaps"] = generate_gaze_heatmap(
                sample["gaze_pts"], sigma=self.heatmap_sigma, size=self.heatmap_size
            )

        # add extra dimension to be compatible with temporal model
        for key, item in sample.items():
            if key not in ["path", "dataset", "pids", "gaze_label_texts"]:
                sample[key] = item.unsqueeze(0)

        return sample

    def __len__(self):
        return self.length
