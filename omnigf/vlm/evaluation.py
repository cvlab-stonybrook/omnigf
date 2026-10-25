# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""Evaluation utilities for VLM gaze prediction.

Provides helpers to convert model outputs (text, heatmaps, patch
distributions) into a unified prediction format for metric computation.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from omnigf.utils import spatial_argmax2d
from omnigf.vlm.parser import MultiPersonOutputParser


def extract_predictions(
    approach: str,
    outputs,
    batch: Dict[str, torch.Tensor],
    parser: MultiPersonOutputParser,
    generated_texts: Optional[List[str]] = None,
    pred_heatmaps: Optional[torch.Tensor] = None,
    inout_pred: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Extract predictions from model outputs into a unified format.

    The returned dict always contains at least:

    * ``gaze_pts``: ``[B, N, 2]`` — predicted gaze points (NaN for outside).
    * ``inout_prob``: ``[B, N]`` — P(inside) in ``[0, 1]``.

    Depending on the approach, it may also contain:

    * ``heatmap``: ``[B, N, 64, 64]`` — predicted heatmaps (decoder output).

    Args:
        approach: ``"lm"`` or ``"heatmap"``.
        outputs: VLM forward outputs (for LM loss extraction).
        batch: Collated batch dict from :class:`~omnigf.vlm.collator.VLMGazeCollator`.
        parser: Parser for decoding generated text.
        generated_texts: Decoded text strings (required for ``"lm"`` eval).
        pred_heatmaps: ``[B, N, 64, 64]`` from a heatmap decoder.
        inout_pred: ``[B, N]`` in/out logits (Approach 3C).

    Returns:
        Dictionary of prediction tensors.
    """
    B = batch["inout"].size(0)
    N = batch["inout"].size(1)
    device = batch["inout"].device
    result: Dict[str, torch.Tensor] = {}

    # ---- Parse text predictions (available for all approaches) ----
    if generated_texts is not None and "person_ids" in batch:
        gaze_pts_text = torch.full((B, N, 2), float("nan"), device=device)
        inout_prob_text = torch.zeros(B, N, device=device)
        categories_text: List[List[Optional[str]]] = [
            [None] * N for _ in range(B)
        ]

        # ``image_path`` is a Python list of strings carried through by
        # ``VLMGazeCollator`` (one per sample); fall back to a generic
        # marker so the warning still identifies which batch index produced
        # the malformed text when path metadata isn't available.
        image_paths = batch.get("image_path", None)
        for bi, text in enumerate(generated_texts):
            ctx = (
                image_paths[bi]
                if image_paths is not None and bi < len(image_paths)
                else f"batch_idx={bi}"
            )
            preds = parser.parse(text, context=ctx)
            gt_pids = batch["person_ids"][bi]
            for pred in preds:
                pid = pred["pid"]
                pid_mask = gt_pids == pid
                if pid_mask.sum() == 0:
                    continue
                idx = pid_mask.nonzero(as_tuple=True)[0][0].item()
                if "inside" in pred and pred["inside"]:
                    gaze_pts_text[bi, idx, 0] = pred["x"]
                    gaze_pts_text[bi, idx, 1] = pred["y"]
                    inout_prob_text[bi, idx] = 1.0
                    if "category" in pred:
                        categories_text[bi][idx] = pred["category"]
                else:
                    gaze_pts_text[bi, idx, 0] = 0.5
                    gaze_pts_text[bi, idx, 1] = 0.5
                    inout_prob_text[bi, idx] = 0.0

        result["gaze_pts_text"] = gaze_pts_text
        result["inout_prob_text"] = inout_prob_text
        result["categories_text"] = categories_text

    if approach == "lm":
        # Text predictions are the primary output for LM-only
        result["gaze_pts"] = result["gaze_pts_text"]
        result["inout_prob"] = result["inout_prob_text"]

    elif approach == "heatmap":
        assert pred_heatmaps is not None
        assert inout_pred is not None
        inout_prob = torch.sigmoid(inout_pred)
        gaze_pts = _heatmap_to_point(pred_heatmaps)

        result["gaze_pts"] = gaze_pts
        result["inout_prob"] = inout_prob
        result["heatmap"] = pred_heatmaps

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _heatmap_to_point(heatmap: torch.Tensor) -> torch.Tensor:
    """Hard argmax on ``[B, N, H, W]`` heatmap → ``[B, N, 2]`` normalised point.

    Uses :func:`~omnigf.utils.spatial_argmax2d`, consistent with the MTGS
    model's evaluation pipeline.
    """
    B, N, H, W = heatmap.shape
    pts = spatial_argmax2d(heatmap.reshape(B * N, H, W), normalize=True)
    return pts.view(B, N, 2)
