# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""Sanity checks for the VLM gaze data pipeline.

Verifies that:
1. Collated batch tensors have the expected shapes and value ranges.
2. Decoded input_ids reproduce the original prompt and target text.
3. Coordinates in the decoded text match the metadata tensors (gaze_pts,
   head_bboxes, inout) within quantisation tolerance.
4. Label masking is correct: prompt tokens are -100, target tokens are real IDs.

Usage::

    python -m omnigf.tools.vlm_dataset_sanity \\
        --config-path /path/to/vlm_config.yaml \\
        --split train --num_batches 5
"""

import argparse
import re
import sys
from collections import defaultdict

import torch
from omegaconf import OmegaConf

from omnigf.vlm.dataset import VLMGazeDataModule


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _assert_shape(name: str, tensor: torch.Tensor, expected_dims: int):
    assert tensor.ndim == expected_dims, (
        f"{name}: expected {expected_dims}D, got {tensor.ndim}D {tuple(tensor.shape)}"
    )


def _assert_in_range(name: str, tensor: torch.Tensor, low: float, high: float):
    if tensor.numel() == 0:
        return
    if ((tensor < low) | (tensor > high)).any():
        vmin, vmax = tensor.min().item(), tensor.max().item()
        raise ValueError(
            f"{name} has values out of [{low}, {high}]: min={vmin}, max={vmax}"
        )


def _parse_target_line(line: str):
    """Parse a target line like 'P0: INSIDE 450 620' or 'P1: OUTSIDE'.

    Returns (pid, inout, x, y) where x/y are None for OUTSIDE.
    """
    m = re.match(r"P(\d+):\s+INSIDE\s+(\d+)\s+(\d+)", line)
    if m:
        return int(m.group(1)), 1, int(m.group(2)), int(m.group(3))
    m = re.match(r"P(\d+):\s+OUTSIDE", line)
    if m:
        return int(m.group(1)), 0, None, None
    return None


def _parse_prompt_bbox(line: str):
    """Parse a prompt bbox line like 'P0: [450,620,500,670]'.

    Returns (pid, [x1, y1, x2, y2]) or None.
    """
    m = re.match(r"P(\d+):\s+\[(\d+),(\d+),(\d+),(\d+)\]", line)
    if m:
        return int(m.group(1)), [int(m.group(i)) for i in range(2, 6)]
    return None


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sanity check for VLMGazeDataModule pipeline."
    )
    parser.add_argument(
        "--config-path", type=str, required=True,
        help="Path to vlm_config.yaml",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "val", "test"],
    )
    parser.add_argument("--num_batches", type=int, default=5)
    parser.add_argument(
        "--coord_tol", type=int, default=2,
        help="Tolerance (in quantised bins) for coordinate matching",
    )
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config_path)

    # Ensure num_people is a dict (config may store a scalar).
    if isinstance(cfg.data.num_people, int):
        np_val = cfg.data.num_people
        cfg.data.num_people = {"train": np_val, "val": np_val, "test": "all"}

    dm = VLMGazeDataModule(cfg)

    # Setup and get dataloader
    if args.split == "train":
        dm.setup("fit")
        loader = dm.train_dataloader()
    elif args.split == "val":
        dm.setup("fit")
        loader = dm.val_dataloader()
    else:
        dm.setup("test")
        loader = dm.test_dataloader()

    tokenizer = dm.tokenizer
    coord_bins = dm.coord_bins
    max_people = dm.max_people

    stats = defaultdict(int)

    for bi, batch in enumerate(loader):
        if bi >= args.num_batches:
            break

        B = batch["input_ids"].size(0)

        # ---- Shape checks ----
        _assert_shape("input_ids", batch["input_ids"], 2)
        _assert_shape("attention_mask", batch["attention_mask"], 2)
        _assert_shape("labels", batch["labels"], 2)
        _assert_shape("inout", batch["inout"], 2)
        _assert_shape("gaze_pts", batch["gaze_pts"], 3)
        _assert_shape("head_bboxes", batch["head_bboxes"], 3)
        _assert_shape("num_annotated", batch["num_annotated"], 1)

        assert batch["input_ids"].shape == batch["labels"].shape, (
            f"input_ids {batch['input_ids'].shape} != labels {batch['labels'].shape}"
        )
        assert batch["input_ids"].shape == batch["attention_mask"].shape
        assert batch["inout"].shape[1] == max_people, (
            f"inout dim 1 = {batch['inout'].shape[1]}, expected max_people={max_people}"
        )
        assert batch["gaze_pts"].shape[1] == max_people
        assert batch["gaze_pts"].shape[2] == 2
        assert batch["head_bboxes"].shape[1] == max_people
        assert batch["head_bboxes"].shape[2] == 4

        # ---- Value range checks ----
        inout = batch["inout"]
        valid_inout = (inout == -1) | (inout == 0) | (inout == 1)
        assert valid_inout.all(), (
            f"inout has invalid values: {inout[~valid_inout].unique().tolist()}"
        )

        gaze_pts = batch["gaze_pts"]
        gaze_valid = gaze_pts[..., 0] >= 0
        if gaze_valid.any():
            _assert_in_range("gaze_pts (valid)", gaze_pts[gaze_valid], 0.0, 1.0)

        bbox = batch["head_bboxes"]
        bbox_valid = bbox.sum(-1) > 0
        if bbox_valid.any():
            # Small negative values are expected from crop/jitter augmentations;
            # the prompt builder clamps to [0,1] before quantising.
            _assert_in_range("head_bboxes (valid)", bbox[bbox_valid], -0.05, 1.05)

        # ---- Label masking check ----
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        for si in range(B):
            ids = input_ids[si]
            lbl = labels[si]

            # Where labels are not -100, they must match input_ids
            target_mask = lbl != -100
            if target_mask.any():
                mismatches = (ids[target_mask] != lbl[target_mask]).sum().item()
                assert mismatches == 0, (
                    f"Batch {bi} sample {si}: {mismatches} target token(s) "
                    f"differ between input_ids and labels"
                )

            # At least some tokens should be masked (prompt) and some unmasked (target)
            n_masked = (lbl == -100).sum().item()
            n_target = target_mask.sum().item()
            assert n_masked > 0, (
                f"Batch {bi} sample {si}: no prompt tokens masked"
            )
            assert n_target > 0, (
                f"Batch {bi} sample {si}: no target tokens (all -100)"
            )

        # ---- Decode and cross-check with metadata ----
        for si in range(B):
            n_ann = batch["num_annotated"][si].item()
            decoded = tokenizer.decode(input_ids[si], skip_special_tokens=False)

            # Extract target portion (labels != -100)
            lbl = labels[si]
            target_ids = input_ids[si][lbl != -100]
            target_text = tokenizer.decode(target_ids, skip_special_tokens=True)

            # Parse target lines and verify against metadata
            for line in target_text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                parsed = _parse_target_line(line)
                if parsed is None:
                    print(f"  [WARN] Batch {bi} sample {si}: "
                          f"unparseable target line: {line!r}")
                    stats["unparseable_lines"] += 1
                    continue

                pid, io, tx, ty = parsed

                # Check pid is in range
                assert 0 <= pid < n_ann, (
                    f"Batch {bi} sample {si}: target P{pid} "
                    f"out of range [0, {n_ann})"
                )

                # Check inout matches metadata
                meta_io = batch["inout"][si, pid].item()
                assert int(meta_io) == io, (
                    f"Batch {bi} sample {si} P{pid}: "
                    f"target inout={io} != metadata inout={int(meta_io)}"
                )

                # Check gaze coordinates match (within quantisation tolerance)
                if io == 1 and tx is not None:
                    meta_gx = batch["gaze_pts"][si, pid, 0].item()
                    meta_gy = batch["gaze_pts"][si, pid, 1].item()
                    expected_x = int(max(0, min(1, meta_gx)) * (coord_bins - 1))
                    expected_y = int(max(0, min(1, meta_gy)) * (coord_bins - 1))
                    assert abs(tx - expected_x) <= args.coord_tol, (
                        f"Batch {bi} sample {si} P{pid}: "
                        f"target x={tx} != expected {expected_x} "
                        f"(meta_gx={meta_gx:.4f})"
                    )
                    assert abs(ty - expected_y) <= args.coord_tol, (
                        f"Batch {bi} sample {si} P{pid}: "
                        f"target y={ty} != expected {expected_y} "
                        f"(meta_gy={meta_gy:.4f})"
                    )

                stats[f"inout={io}"] += 1

            # Parse prompt bboxes and verify against metadata
            prompt_ids = input_ids[si][lbl == -100]
            prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            for line in prompt_text.strip().split("\n"):
                parsed = _parse_prompt_bbox(line.strip())
                if parsed is None:
                    continue
                pid, bbox_q = parsed
                if pid >= n_ann:
                    continue  # unknown person in "include_marked" mode

                meta_bbox = batch["head_bboxes"][si, pid]
                expected_bbox = (meta_bbox.clamp(0, 1) * (coord_bins - 1)).int().tolist()
                for dim_name, got, exp in zip(
                    ["x1", "y1", "x2", "y2"], bbox_q, expected_bbox
                ):
                    assert abs(got - exp) <= args.coord_tol, (
                        f"Batch {bi} sample {si} P{pid} bbox {dim_name}: "
                        f"prompt={got} != expected={exp}"
                    )

            stats["samples_checked"] += 1

        # ---- Print batch summary ----
        print(
            f"Batch {bi}: B={B}, L={input_ids.shape[1]}, "
            f"pixel_values={tuple(batch['pixel_values'].shape)}, "
            f"num_annotated={batch['num_annotated'].tolist()}"
        )

        # Print decoded sample 0 for visual inspection
        if bi == 0:
            lbl = labels[0]
            target_ids = input_ids[0][lbl != -100]
            prompt_ids = input_ids[0][lbl == -100]
            print("\n--- Sample 0 prompt (decoded) ---")
            print(tokenizer.decode(prompt_ids, skip_special_tokens=True)[:500])
            print("\n--- Sample 0 target (decoded) ---")
            print(tokenizer.decode(target_ids, skip_special_tokens=True))
            print()

    print("\n" + "=" * 60)
    print("Summary:")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print("=" * 60)
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()
