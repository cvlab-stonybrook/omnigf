# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from typing import Dict, List, Tuple, Union, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset

from omnigf.datasets.childplay_temporal import ChildPlayDataset_temporal
from omnigf.datasets.gazefollow import GazeFollowDataset
from omnigf.datasets.gazehoi import GazeHOIDataset
from omnigf.datasets.goo_real import GOORealDataset
from omnigf.datasets.videoattentiontarget_temporal import VideoAttentionTargetDataset_temporal
from omnigf.datasets.videocoatt_temporal import VideoCoAttDataset_temporal
from omnigf.datasets.uco_laeo_temporal import VideoLAEODataset_temporal
from omnigf.train.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomCropSafeGaze,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from omnigf.utils import Stage, pair
from omnigf.utils.image import IMG_MEAN, IMG_STD


class UnifiedGazeFrameDataset(Dataset):
    """Unified frame-level dataset view across GazeFollow, VAT and ChildPlay.

    The wrapped source datasets already implement the heavy lifting (loading, bbox jitter,
    crops and label generation). This class creates a deterministic flattened index:
    (source_dataset, local_sample_idx), then converts each sample to a frame-level contract.
    """

    VISIBILITY_INSIDE = 1
    VISIBILITY_OUTSIDE = 0
    VISIBILITY_UNKNOWN = -1

    def __init__(self, datasets: Dict[str, Dataset]):
        super().__init__()
        self.datasets = datasets
        self.sources = list(datasets.keys())
        self.index: List[Tuple[str, int]] = []
        for source in self.sources:
            source_len = len(self.datasets[source])
            self.index.extend([(source, i) for i in range(source_len)])

    def __len__(self) -> int:
        return len(self.index)

    def get_source_name(self, index: int) -> str:
        source, _ = self.index[index]
        return source

    def _select_center_frame(self, sample: Dict) -> Dict:
        frame_sample = {}
        for key, item in sample.items():
            if key in ["dataset", "path", "pids"]:
                # "pids" is not temporal-stacked in source datasets (1D [N]),
                # so it must not go through center-frame selection.
                frame_sample[key] = item
                continue
            if isinstance(item, torch.Tensor) and item.ndim >= 1:
                # Temporal datasets have T first; GazeFollow has T=1 by construction.
                center_idx = item.shape[0] // 2
                frame_sample[key] = item[center_idx].unsqueeze(0)
            else:
                frame_sample[key] = item

        # Keep path format compatible with existing downstream code.
        path = frame_sample["path"]
        if isinstance(path, list):
            if len(path) > 0:
                frame_sample["path"] = [path[len(path) // 2]]
            else:
                frame_sample["path"] = [""]
        else:
            frame_sample["path"] = [str(path)]
        return frame_sample

    def __getitem__(self, index: int) -> Dict:
        source, source_index = self.index[index]
        sample = self.datasets[source][source_index]
        frame_sample = self._select_center_frame(sample)

        # Canonical visibility key for unified access.
        frame_sample["visibility"] = frame_sample["inout"].clone()
        frame_sample["dataset"] = source

        # Normalise person ID key: all source datasets use "pids".
        # GazeFollow wraps it with unsqueeze(0) → [1, N]; temporal datasets
        # keep it 1D [N] (skipped from temporal stacking).  Flatten to [N].
        if "pids" in frame_sample:
            pids = frame_sample.pop("pids")
            frame_sample["person_ids"] = pids.squeeze(0) if pids.ndim > 1 else pids

        return frame_sample


class UnifiedGazeFrameDataModule(pl.LightningDataModule):
    """Extended data module supporting all MTGS datasets via a registry.

    Supports GazeFollow, VAT, ChildPlay, VideoCoAtt and UCO-LAEO.  Adding a
    new dataset only requires a new entry in :attr:`DATASET_REGISTRY`.
    """

    # Registry: maps source name → (dataset class, root key, temporal flag,
    # available social labels).
    DATASET_REGISTRY: Dict[str, dict] = {
        "gazefollow": {
            "cls": GazeFollowDataset,
            "root_key": "root_gazefollow",
            "temporal": False,
            "social_labels": ["lah"],
        },
        "vat": {
            "cls": VideoAttentionTargetDataset_temporal,
            "root_key": "root_vat",
            "temporal": True,
            "social_labels": ["lah"],
        },
        "childplay": {
            "cls": ChildPlayDataset_temporal,
            "root_key": "root_childplay",
            "temporal": True,
            "social_labels": ["lah"],
        },
        "videocoatt": {
            "cls": VideoCoAttDataset_temporal,
            "root_key": "root_videocoatt",
            "temporal": True,
            "social_labels": ["lah", "laeo", "coatt"],
        },
        "uco_laeo": {
            "cls": VideoLAEODataset_temporal,
            "root_key": "root_uco_laeo",
            "temporal": True,
            "social_labels": ["lah", "laeo"],
        },
        # GazeHOI is per-pair flattened (one (head, target) row per
        # __getitem__), single-frame, semantic-only — no social labels.
        "gazehoi": {
            "cls": GazeHOIDataset,
            "root_key": "root_gazehoi",
            "temporal": False,
            "social_labels": [],
        },
        # GOO-Real: retail-store gaze following, single-person, single-frame.
        # Cross-dataset evaluation only (no social labels, not wired for
        # training). Annotations live in a separate pickle dir (``goo_ann_root``).
        "goo_real": {
            "cls": GOORealDataset,
            "root_key": "root_goo_real",
            "temporal": False,
            "social_labels": [],
        },
    }

    # Per-source, per-split frame sampling stride defaults.
    STRIDE_DEFAULTS: Dict[str, Dict[str, int]] = {
        "vat": {"train": 6, "val": 6, "test": 1},
        "childplay": {"train": 6, "val": 6, "test": 1},
        "videocoatt": {"train": 6, "val": 6, "test": 1},
        "uco_laeo": {"train": 6, "val": 6, "test": 1},
    }

    def __init__(
        self,
        ann_root: str,
        batch_size: Union[int, dict],
        image_size: Tuple[int, int],
        head_size: Tuple[int, int],
        heatmap_size: int,
        num_people: Union[int, str, Dict[str, Union[int, str]]],
        include_sources: Dict[str, bool],
        # Preferred: single dict of dataset roots.
        dataset_roots: Optional[Dict[str, str]] = None,
        source_sampling: Optional[Dict[str, float]] = None,
        max_train_samples: Union[int, None] = None,
        max_val_samples: Union[int, None] = None,
        max_test_samples: Union[int, None] = None,
        # Optional separate source selection for val/test splits.
        # When None, falls back to include_sources.
        val_include_sources: Optional[Dict[str, bool]] = None,
        test_include_sources: Optional[Dict[str, bool]] = None,
        # Backward-compatible individual root params.
        root_gazefollow: Optional[str] = None,
        root_vat: Optional[str] = None,
        root_childplay: Optional[str] = None,
        root_videocoatt: Optional[str] = None,
        root_uco_laeo: Optional[str] = None,
        root_gazehoi: Optional[str] = None,
        skip_normalize: bool = True,
        gf_include_inout: bool = False,
        gf_use_semgaze_bbox: bool = False,
        gf_semgaze_ann_root: Optional[str] = None,
        gf_use_semantic_labels: bool = False,
        gf_semantic_vocab_dir: Optional[str] = None,
        gf_train_prediction_json: Optional[str] = None,
        # GazeHOI uses its own h5 directory (separate from VSGaze
        # ``ann_root``) and a single shared CLIP-templated vocab.
        gazehoi_ann_root: Optional[str] = None,
        gazehoi_vocab2id_path: Optional[str] = None,
        # GOO-Real stores its annotations as pickles in a separate directory
        # (separate from VSGaze ``ann_root``). Eval-only cross-dataset source.
        goo_ann_root: Optional[str] = None,
        # When set, train-split h5 files for the listed sources are loaded
        # from ``<source>_train_withpseudo.h5`` instead of the standard
        # ``<source>_train.h5``. Sources not listed (or set to False) load
        # the standard files. Always applies to train split only.
        pseudo_train_sources: Optional[Dict[str, bool]] = None,
    ):
        super().__init__()
        self.ann_root = ann_root
        self.pseudo_train_sources = dict(pseudo_train_sources or {})
        self.gf_use_semgaze_bbox = gf_use_semgaze_bbox
        self.gf_semgaze_ann_root = gf_semgaze_ann_root
        self.gf_use_semantic_labels = gf_use_semantic_labels
        self.gf_semantic_vocab_dir = gf_semantic_vocab_dir
        # Train-only override of GazeFollow GT gaze with saved predictions.
        # When set, the train split's GazeFollow source loads this JSON
        # (saved by ``scripts/save_vlm_predictions.py``) and replaces GT
        # gaze in __getitem__ before shuffle/jitter/transform.
        self.gf_train_prediction_json = gf_train_prediction_json
        self.skip_normalize = skip_normalize

        # Build dataset_roots from either the dict or individual params.
        if dataset_roots is not None:
            self.dataset_roots = dict(dataset_roots)
        else:
            self.dataset_roots = {}
        # Individual root_* params fill in missing entries (backward compat).
        _compat = {
            "root_gazefollow": root_gazefollow,
            "root_vat": root_vat,
            "root_childplay": root_childplay,
            "root_videocoatt": root_videocoatt,
            "root_uco_laeo": root_uco_laeo,
            "root_gazehoi": root_gazehoi,
        }
        for key, val in _compat.items():
            if val is not None and key not in self.dataset_roots:
                self.dataset_roots[key] = val

        if isinstance(image_size, ListConfig):
            image_size = OmegaConf.to_object(image_size)
        self.image_size = pair(image_size)
        self.head_size = head_size
        self.heatmap_sigma = int(np.mean(heatmap_size) * 3 / 64)
        self.heatmap_size = heatmap_size
        self.num_people = num_people
        self.batch_size = (
            {stage: batch_size for stage in Stage}
            if isinstance(batch_size, int)
            else batch_size
        )
        self.include_sources = include_sources
        self.val_include_sources = val_include_sources
        self.test_include_sources = test_include_sources
        self.source_sampling = source_sampling or {}
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples
        self.gf_include_inout = gf_include_inout
        self.gazehoi_ann_root = gazehoi_ann_root
        self.gazehoi_vocab2id_path = gazehoi_vocab2id_path
        self.goo_ann_root = goo_ann_root

    def _build_transforms(self, split: str):
        if split == "train":
            ops = [
                RandomCropSafeGaze(
                    aspect=(self.image_size[0] / self.image_size[1]), p=1.0
                ),
                RandomHorizontalFlip(p=0.5),
                ColorJitter(
                    brightness=(0.5, 1.5),
                    contrast=(0.5, 1.5),
                    saturation=(0.0, 1.5),
                    hue=None,
                    p=0.8,
                ),
                Resize(img_size=self.image_size, head_size=self.head_size),
                ToTensor(),
            ]
        else:
            ops = [
                Resize(img_size=self.image_size, head_size=self.head_size),
                ToTensor(),
            ]
        if not self.skip_normalize:
            ops.append(Normalize(img_mean=IMG_MEAN, img_std=IMG_STD))
        return Compose(ops)

    @staticmethod
    def _get_stride(source_name: str, split: str) -> int:
        """Per-source, per-split frame sampling stride."""
        return UnifiedGazeFrameDataModule.STRIDE_DEFAULTS.get(
            source_name, {}
        ).get(split, 1)

    def _build_source_datasets(self, split: str) -> Dict[str, Dataset]:
        datasets = {}
        transform = self._build_transforms(split=split)

        # Use per-split source overrides when available.
        if split == "test" and self.test_include_sources is not None:
            active_sources = self.test_include_sources
        elif split in 'val' and self.val_include_sources is not None:
            active_sources = self.val_include_sources
        else:
            active_sources = self.include_sources

        for source_name, config in self.DATASET_REGISTRY.items():

            if not active_sources.get(source_name, False):
                continue

            root_key = config["root_key"]
            root = self.dataset_roots.get(root_key)
            if root is None:
                raise ValueError(
                    f"Root path for '{source_name}' (key='{root_key}') not "
                    f"configured.  Pass it via dataset_roots or root_{source_name}."
                )

            # Common kwargs shared by all datasets.
            # ``num_people`` may be a scalar (int / "all") shared by all
            # splits, or a per-split dict. The per-split dict is what lets
            # val/test run with ``"all"`` (keep every real person in the
            # frame) while train stays at a scalar cap — avoiding the
            # random ``num_keep`` truncation that silently drops real pids
            # and leaks aux target-object pids at eval time.
            np_val = self.num_people
            if isinstance(np_val, (dict, DictConfig)):
                np_val = np_val.get(split, np_val.get("train"))
            kwargs = {
                "root": root,
                "ann_root": self.ann_root,
                "split": split,
                "transform": transform,
                "tr": (-0.1, 0.1) if split == "train" else (0.0, 0.0),
                "heatmap_size": self.heatmap_size,
                "heatmap_sigma": self.heatmap_sigma,
                "num_people": np_val,
            }
            if source_name == "gazefollow":
                kwargs["include_inout"] = self.gf_include_inout
                kwargs["use_semgaze_bbox"] = self.gf_use_semgaze_bbox
                kwargs["semgaze_ann_root"] = self.gf_semgaze_ann_root
                kwargs["use_semantic_labels"] = self.gf_use_semantic_labels
                kwargs["semantic_vocab_dir"] = self.gf_semantic_vocab_dir
                if split == "train":
                    kwargs["prediction_json"] = self.gf_train_prediction_json
            if source_name == "gazehoi":
                # GazeHOI has its own h5 dir; override the shared
                # ``ann_root`` so the dataset reads ``gazehoi_{split}.h5``
                # from the right location.
                if self.gazehoi_ann_root is not None:
                    kwargs["ann_root"] = self.gazehoi_ann_root
                kwargs["use_semantic_labels"] = True
                kwargs["vocab2id_path"] = self.gazehoi_vocab2id_path
            if source_name == "goo_real":
                # GOO-Real annotations are pickles in their own directory;
                # override the shared ``ann_root`` so the dataset reads
                # ``<split>realhumansNew.pickle`` from the right location.
                if self.goo_ann_root is not None:
                    kwargs["ann_root"] = self.goo_ann_root
            # Temporal datasets need additional parameters.
            if config["temporal"]:
                stride = self._get_stride(source_name, split)
                kwargs.update(
                    {
                        "stride": stride,
                        "temporal_context": 0,
                        "temporal_stride": stride,
                        "image_size": self.image_size,
                        "head_size": self.head_size,
                    }
                )

            # Opt-in to the pseudo-annotated h5 file (train split only).
            if (
                split == "train"
                and source_name in ("videocoatt", "uco_laeo")
                and self.pseudo_train_sources.get(source_name, False)
            ):
                kwargs["ann_filename_suffix"] = "_withpseudo_cleaned"

            datasets[source_name] = config["cls"](**kwargs)

        if len(datasets) == 0:
            raise ValueError("No source datasets were enabled for unified_gaze_frame.")
        return datasets

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = UnifiedGazeFrameDataset(
                self._build_source_datasets(split="train")
            )
            self.val_dataset = UnifiedGazeFrameDataset(
                self._build_source_datasets(split="val")
            )
        elif stage == "validate":
            self.val_dataset = UnifiedGazeFrameDataset(
                self._build_source_datasets(split="val")
            )
        elif stage == "test":
            self.test_dataset = UnifiedGazeFrameDataset(
                self._build_source_datasets(split="test")
            )

    def _maybe_subset(self, dataset, max_samples):
        if max_samples is None:
            return dataset
        return Subset(dataset, range(min(max_samples, len(dataset))))

    def _build_weighted_sampler(self, dataset: UnifiedGazeFrameDataset):
        if not self.source_sampling:
            return None
        weights = []
        for i in range(len(dataset)):
            source = dataset.get_source_name(i)
            weights.append(float(self.source_sampling.get(source, 1.0)))
        weights = torch.tensor(weights, dtype=torch.double)
        return WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )

    def train_dataloader(self):
        train_dataset = self._maybe_subset(self.train_dataset, self.max_train_samples)
        sampler = None
        shuffle = True
        if isinstance(train_dataset, UnifiedGazeFrameDataset):
            sampler = self._build_weighted_sampler(train_dataset)
            if sampler is not None:
                shuffle = False

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size[Stage.TRAIN],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=8,
            pin_memory=False,
        )

    def val_dataloader(self):
        val_dataset = self._maybe_subset(self.val_dataset, self.max_val_samples)
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size[Stage.VAL],
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )

    def test_dataloader(self):
        test_dataset = self._maybe_subset(self.test_dataset, self.max_test_samples)
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size[Stage.TEST],
            shuffle=False,
            num_workers=4,
            pin_memory=False,
        )
