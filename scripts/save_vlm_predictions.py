# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""Save VLM gaze predictions to a JSON file for downstream evaluation.

Runs the trained VLM gaze model on the test set and saves per-sample
predictions (gaze coordinates, in/out, person IDs, head bboxes) along
with ground-truth semantic labels.  The output JSON can be consumed by
``eval_zeroshot_cot_semantic.py`` for zero-shot Chain-of-Thought semantic
evaluation.

Usage::

    cd scripts/
    python save_vlm_predictions.py \\
        test.checkpoint=/path/to/checkpoint.ckpt \\
        vlm.approach=heatmap \\
        +save_predictions_path=/path/to/output.json
"""

import json
import logging
import math
import os
from typing import Dict, List

import hydra
import torch
import lightning.pytorch as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from omnigf.vlm.checkpoint import load_checkpoint
from omnigf.vlm.models_ge import VLMGazeModelGE
from omnigf.vlm.dataset import VLMGazeDataModule
from omnigf.vlm.evaluation import extract_predictions
from omnigf.performance.metrics import Distance, AUC, GFTestDistance, GFTestAUC
from sklearn.metrics import average_precision_score

logger = logging.getLogger("SavePredictions")


def nan_safe(val):
    """Convert NaN/Inf floats to None for JSON serialisation."""
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def tensor_to_list(t):
    """Convert tensor to JSON-safe nested list."""
    raw = t.cpu().tolist()
    if isinstance(raw, list):
        return [[nan_safe(v) for v in row] if isinstance(row, list) else nan_safe(row) for row in raw]
    return nan_safe(raw)


@hydra.main(
    config_path="./../omnigf/config/",
    config_name="vlm_config.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    save_path = cfg.get("save_predictions_path", None)
    if save_path is None:
        save_path = os.path.join(
            cfg.experiment.output_folder, "predictions.json",
        )
    logger.info("Will save predictions to: %s", save_path)

    # Precision and seed
    torch.set_float32_matmul_precision(cfg.train.matmul_precision)
    if cfg.train.seed is not None:
        pl.seed_everything(cfg.train.seed)

    # Which split to run inference on (default: test).
    # Use hydra override: +inference.split=train|val|test
    split = "test"
    if "inference" in cfg and "split" in cfg.inference:
        split = cfg.inference.split
    logger.info("Target split: %s", split)

    # Optional: override the temporal-source train stride before the data
    # module builds the underlying datasets. The default
    # ``UnifiedGazeFrameDataModule.STRIDE_DEFAULTS["<src>"]["train"] = 6``
    # samples one frame per 6 — fast and non-redundant, but at stage-2
    # train-time the frame jitter (``±temporal_stride//2`` per
    # e.g. childplay_temporal.py:110-136) lands on neighbouring frames
    # whose paths are NOT in the saved JSON. Setting train_stride=1 here
    # dumps a prediction for every train frame so the path-keyed lookup
    # at stage-2 always hits regardless of the jitter range.
    train_stride = None
    if "inference" in cfg and "train_stride" in cfg.inference:
        train_stride = int(cfg.inference.train_stride)
    if split == "train" and train_stride is not None:
        from omnigf.datasets.unified_gaze_frame import UnifiedGazeFrameDataModule
        for _src, _splits in UnifiedGazeFrameDataModule.STRIDE_DEFAULTS.items():
            _splits["train"] = train_stride
        logger.info(
            "Overrode STRIDE_DEFAULTS[*]['train'] = %d for train-set save.",
            train_stride,
        )

    # Build data module
    data = VLMGazeDataModule(cfg)

    if split == "test":
        data.setup("test")
        target_dataset = data.test_dataset
    elif split in ("train", "val"):
        data.setup("fit")
        target_dataset = (
            data.train_dataset if split == "train" else data.val_dataset
        )
        if split == "train":
            # Make the train split deterministic: disable in-source
            # augmentation (bbox jitter, people shuffle) and use val-style
            # transforms. This is required so that the per-index saved
            # prediction is stable and matches what SemanticCoTDataset
            # will look up by index at training time.
            inner = data._inner_dm
            val_transform = inner._build_transforms(split="val")
            for src_name, src in inner.train_dataset.datasets.items():
                if hasattr(src, "split"):
                    src.split = "val"
                if hasattr(src, "transform"):
                    src.transform = val_transform
            logger.info(
                "Patched %d source dataset(s) to val-mode for deterministic "
                "train-set inference.", len(inner.train_dataset.datasets),
            )
            # The outer VLMGazeDataset wrapper still has split="train",
            # which short-circuits the social_gaze_mode keep-all-real-
            # persons branch in omnigf/vlm/dataset.py (gate requires
            # ``self.split in ("val", "test")``). Flip the wrapper split
            # to "val" so an inference run on the train split also keeps
            # ``inout=-1`` persons in the prompt when
            # ``vlm.social_gaze_mode=True``.
            if getattr(data.train_dataset, "social_gaze_mode", False):
                data.train_dataset.split = "val"
                logger.info(
                    "Patched VLMGazeDataset.split='val' for train-set "
                    "inference (keeps inout=-1 persons under "
                    "social_gaze_mode=True).",
                )
    else:
        raise ValueError(f"Unknown inference split: {split!r}")
    logger.info("Dataset size: %d", len(target_dataset))

    # Build model
    model = VLMGazeModelGE(cfg)

    # Load checkpoint
    test_ckpt = getattr(cfg.test, "checkpoint", None)
    if test_ckpt:
        logger.info("Loading checkpoint: %s", test_ckpt)
        missing, unexpected = load_checkpoint(model, test_ckpt)
        if missing:
            logger.warning("Missing keys: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected)
        logger.info("Checkpoint loaded successfully")
    else:
        logger.warning("No test.checkpoint specified — using random weights!")

    device = torch.device("cuda" if cfg.device == "cuda" else "cpu")
    model = model.to(device).eval()

    tokenizer = data.tokenizer

    # Build a deterministic DataLoader for the chosen split.
    # ``data.collator`` is built with ``max_people=cfg.vlm.max_people``
    # (typically 5) and would silently truncate frames with more real
    # persons — fine for training, but at inference we want predictions
    # for every real person in the prompt. Mirror what
    # ``VLMGazeDataModule.test_dataloader`` does (omnigf/vlm/dataset.py:935)
    # and build a fresh collator with ``max_people=None``, which falls
    # back to per-batch-max padding.
    from omnigf.vlm.collator import VLMGazeCollator
    inference_collator = VLMGazeCollator(
        processor=data.processor,
        approach=data.approach,
        max_people=None,
        use_gaze_encoder=data.use_gaze_encoder,
        mask_point_block_loss=data.mask_point_block_loss,
        prompt_mode=getattr(cfg.vlm, "prompt_mode", "point"),
    )
    test_batch_size = getattr(cfg.test, "batch_size", 1)
    test_loader = DataLoader(
        target_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=getattr(cfg.data, "num_workers", 4),
        collate_fn=inference_collator,
        pin_memory=False,
        drop_last=False,
    )

    coord_bins = getattr(cfg.vlm, "coord_bins", 1000)
    records: List[Dict] = []
    skipped_empty_batches = 0

    # ---- Metrics ----
    gf_dist = GFTestDistance().to(device)
    gf_auc = GFTestAUC().to(device)
    gf_dist_text = GFTestDistance().to(device)
    per_dist = Distance().to(device)
    per_auc = AUC().to(device)
    per_dist_text = Distance().to(device)
    inout_gt_all: List[torch.Tensor] = []
    inout_pred_all: List[torch.Tensor] = []
    inout_valid_all: List[torch.Tensor] = []

    logger.info("Running inference on %d batches...", len(test_loader))

    with torch.no_grad(), torch.cuda.amp.autocast(enabled="bf16" in str(cfg.train.precision)):
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Saving predictions")):
            # ``VLMGazeDataset.__getitem__`` returns None for a frame with no
            # annotated persons, and ``VLMGazeCollator`` returns None when the
            # WHOLE batch is empty after dropping those. VLMGazeModelGE's
            # val/test steps already short-circuit on this; do the same here.
            #
            # ChildPlay test has 52/9986 such frames (0.52%), and they cluster
            # inside clips (runs of up to 15 consecutive), so whether this
            # triggers depends on the batch size: bs 1/4/8 hit an entirely-empty
            # batch, bs >= 16 does not. VAT test has none, which is why only
            # ChildPlay crashed. Skipping loses nothing -- a frame with no
            # annotated person has nothing to score.
            if batch is None:
                skipped_empty_batches += 1
                continue

            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Run inference
            preds = model.forward(batch, tokenizer=tokenizer)

            B = batch["inout"].size(0)
            N = batch["inout"].size(1)
            is_gazefollow = "gaze_pts_all" in batch

            # ---- Update gaze metrics ----
            if is_gazefollow:
                last_idx = batch["num_annotated"] - 1
                bi_range = torch.arange(B, device=last_idx.device)
                gaze_pred = preds["gaze_pts"][bi_range, last_idx]
                gaze_gt_all = batch["gaze_pts_all"]
                gf_dist.update(gaze_pred, gaze_gt_all)
                if "heatmap" in preds:
                    hm_pred = preds["heatmap"][bi_range, last_idx]
                    gf_auc.update(hm_pred, gaze_gt_all)
                if "gaze_pts_text" in preds:
                    gaze_pred_text = preds["gaze_pts_text"][bi_range, last_idx]
                    gf_dist_text.update(gaze_pred_text, gaze_gt_all)
            else:
                gaze_pred = preds["gaze_pts"]
                gaze_gt = batch["gaze_pts"]
                inout_gt = batch["inout"]
                per_dist.update(gaze_pred, gaze_gt, inout_gt)
                if "heatmap" in preds and "gaze_heatmaps" in batch:
                    per_auc.update(
                        preds["heatmap"].view(-1, 64, 64),
                        batch["gaze_heatmaps"].view(-1, 64, 64),
                        inout_gt.view(-1),
                    )
                if "inout_prob" in preds:
                    valid = (inout_gt.view(-1) != -1)
                    inout_gt_all.append(inout_gt.view(-1).cpu())
                    inout_pred_all.append(preds["inout_prob"].view(-1).cpu())
                    inout_valid_all.append(valid.cpu())
                if "gaze_pts_text" in preds:
                    per_dist_text.update(preds["gaze_pts_text"], gaze_gt, inout_gt)

            for bi in range(B):
                record = {
                    "batch_idx": batch_idx,
                    "sample_idx": batch_idx * test_loader.batch_size + bi,
                }

                # Image path (passed through by collator as list of str)
                if "image_path" in batch:
                    record["image_path"] = batch["image_path"][bi]

                # Person IDs and head bboxes
                record["person_ids"] = batch["person_ids"][bi].cpu().tolist()
                record["head_bboxes"] = tensor_to_list(batch["head_bboxes"][bi])
                record["num_annotated"] = int(batch["num_annotated"][bi].item())

                # Ground-truth
                record["inout_gt"] = batch["inout"][bi].cpu().tolist()

                if is_gazefollow:
                    record["gaze_pts_all_gt"] = tensor_to_list(batch["gaze_pts_all"][bi])
                else:
                    record["gaze_pts_gt"] = tensor_to_list(batch["gaze_pts"][bi])

                # GT semantic labels (test vocab IDs)
                if "gaze_test_label_ids" in batch:
                    record["gaze_test_label_ids"] = batch["gaze_test_label_ids"][bi].cpu().tolist()
                if "gaze_multi_test_label_ids" in batch:
                    record["gaze_multi_test_label_ids"] = tensor_to_list(
                        batch["gaze_multi_test_label_ids"][bi],
                    )

                # ---- Predictions ----
                # Use heatmap-based predictions if available, otherwise text
                record["gaze_pts_pred"] = tensor_to_list(preds["gaze_pts"][bi])
                record["inout_prob_pred"] = tensor_to_list(preds["inout_prob"][bi])

                # Also save text predictions separately
                if "gaze_pts_text" in preds:
                    record["gaze_pts_text_pred"] = tensor_to_list(preds["gaze_pts_text"][bi])
                if "inout_prob_text" in preds:
                    record["inout_prob_text_pred"] = tensor_to_list(preds["inout_prob_text"][bi])

                # Categories from text generation (if using point_semantic mode)
                if "categories_text" in preds and preds["categories_text"][bi]:
                    record["categories_text_pred"] = preds["categories_text"][bi]

                records.append(record)

    # ---- Compute and log gaze metrics ----
    metrics: Dict[str, float] = {}
    if gf_dist.n_observations > 0:
        avg_dist, min_dist = gf_dist.compute()
        metrics["gf_avg_dist"] = float(avg_dist)
        metrics["gf_min_dist"] = float(min_dist)
    if gf_auc.n_observations > 0:
        metrics["gf_auc"] = float(gf_auc.compute())
    if gf_dist_text.n_observations > 0:
        a_dist, m_dist = gf_dist_text.compute()
        metrics["gf_avg_dist_text"] = float(a_dist)
        metrics["gf_min_dist_text"] = float(m_dist)
    if per_dist.n_observations > 0:
        metrics["dist"] = float(per_dist.compute())
    if per_auc.n_observations > 0:
        metrics["auc"] = float(per_auc.compute())
    if per_dist_text.n_observations > 0:
        metrics["dist_text"] = float(per_dist_text.compute())
    if inout_gt_all:
        gt = torch.cat(inout_gt_all).numpy()
        pr = torch.cat(inout_pred_all).numpy()
        vd = torch.cat(inout_valid_all).numpy().astype(bool)
        if vd.any() and len(set(gt[vd].tolist())) > 1:
            metrics["inout_ap"] = float(average_precision_score(gt[vd], pr[vd]))

    if metrics:
        logger.info("==== Gaze metrics ====")
        for k, v in metrics.items():
            logger.info("  %s: %.4f", k, v)

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(records, f, indent=2)
    logger.info("Saved %d prediction records to %s", len(records), save_path)
    if skipped_empty_batches:
        logger.warning(
            "Skipped %d batch(es) in which EVERY frame had no annotated person. "
            "Those frames contribute nothing to any metric, but the record "
            "count will be below the raw frame count -- state the scored count, "
            "not the split size, when reporting.", skipped_empty_batches)

    # Save metrics alongside predictions
    if metrics:
        metrics_path = os.path.splitext(save_path)[0] + "_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
