# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Quick sanity checks for the unified frame-level gaze dataset."""

import argparse
from collections import defaultdict

import torch

from omnigf.datasets.unified_gaze_frame import UnifiedGazeFrameDataModule
from omnigf.utils import Stage


def _assert_in_range(name: str, tensor: torch.Tensor, low: float, high: float):
    if tensor.numel() == 0:
        return
    if ((tensor < low) | (tensor > high)).any():
        raise ValueError(f"{name} has values out of [{low}, {high}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann_root", type=str, required=True)
    parser.add_argument("--gazefollow_root", type=str, required=True)
    parser.add_argument("--vat_root", type=str, required=True)
    parser.add_argument("--childplay_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=5)
    args = parser.parse_args()

    dm = UnifiedGazeFrameDataModule(
        root_gazefollow=args.gazefollow_root,
        root_vat=args.vat_root,
        root_childplay=args.childplay_root,
        ann_root=args.ann_root,
        batch_size={Stage.TRAIN: args.batch_size, Stage.VAL: args.batch_size, Stage.TEST: 1},
        image_size=(448, 448),
        head_size=(224, 224),
        heatmap_size=64,
        num_people={"train": 4, "val": 4, "test": "all"},
        include_sources={"gazefollow": True, "vat": True, "childplay": True},
    )

    if args.split in ["train", "val"]:
        dm.setup("fit")
        loader = dm.train_dataloader() if args.split == "train" else dm.val_dataloader()
    else:
        dm.setup("test")
        loader = dm.test_dataloader()

    visibility_counts = defaultdict(int)
    source_counts = defaultdict(int)

    for bi, batch in enumerate(loader):
        if bi >= args.num_batches:
            break
        # Shape checks
        assert batch["image"].ndim == 5, "Expected image shape: (B, 1, C, H, W)"
        assert batch["head_bboxes"].shape[-1] == 4
        assert batch["gaze_pts"].shape[-1] == 2
        assert batch["inout"].shape == batch["visibility"].shape

        # Bounds checks for valid values only.
        bbox = batch["head_bboxes"]
        gaze = batch["gaze_pts"]
        inout = batch["inout"]
        bbox_valid = bbox.sum(-1) > 0
        gaze_valid = gaze[..., 0] >= 0
        _assert_in_range("head_bboxes", bbox[bbox_valid], 0.0, 1.0)
        _assert_in_range("gaze_pts", gaze[gaze_valid], 0.0, 1.0)
        if not (((inout == -1) | (inout == 0) | (inout == 1)).all()):
            raise ValueError("inout/visibility labels must be one of {-1, 0, 1}")

        for v in inout.flatten().tolist():
            visibility_counts[int(v)] += 1
        for d in batch["dataset"]:
            source_counts[d] += 1

    print("Visibility counts:", dict(visibility_counts))
    print("Source counts:", dict(source_counts))
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
