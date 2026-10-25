# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""Loss functions for VLM-based gaze prediction.

All loss functions return ``(loss_scalar, logs_dict)`` matching the pattern
used in MTGS's ``train/losses.py``.
"""
import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Approach 3C / 3D: Heatmap MSE
# ---------------------------------------------------------------------------


def compute_heatmap_mse_loss(pred_hm, gt_hm, mask, weight=None):
    """MSE loss on 64x64 gaze heatmaps.

    Follows the same implementation as ``compute_heatmap_loss`` in MTGS's ``train/losses.py``:
    per-element MSE averaged over spatial dims, masked, then averaged over valid persons.

    Args:
        pred_hm: ``[B, N, 64, 64]`` — predicted heatmaps.
        gt_hm: ``[B, N, 64, 64]`` — ground-truth heatmaps.
        mask: ``[B, N]`` boolean — ``True`` for inside-frame persons.
        weight: optional ``[B, N]`` per-person multiplicative weight applied
            on top of ``mask``. Used to down-weight pseudo-labelled persons.

    Returns:
        Tuple of ``(loss, {"heatmap_mse_loss": float})``.
    """
    if mask.sum() == 0:
        zero = torch.tensor(0.0, device=pred_hm.device)
        return zero, {"heatmap_mse_loss": 0.0}

    heatmap_loss = F.mse_loss(pred_hm, gt_hm, reduction="none").mean([2, 3])
    if weight is None:
        heatmap_loss = torch.mul(heatmap_loss, mask)
        loss = torch.sum(heatmap_loss) / torch.sum(mask)
    else:
        eff = mask.float() * weight.to(mask.device)
        heatmap_loss = heatmap_loss * eff
        loss = heatmap_loss.sum() / eff.sum().clamp(min=1e-6)
    return loss, {"heatmap_mse_loss": loss.item()}


# ---------------------------------------------------------------------------
# In/out BCE (shared across approaches)
# ---------------------------------------------------------------------------


def compute_inout_bce_loss(pred_logit, gt_inout, mask, weight=None):
    """Binary cross-entropy for inside/outside classification.

    Follows the same implementation as ``compute_inout_loss`` in MTGS's ``train/losses.py``:
    per-element BCE, masked, then averaged over valid persons.

    Args:
        pred_logit: ``[B, N]`` — raw logits (before sigmoid).
        gt_inout: ``[B, N]`` — ground-truth labels (1=inside, 0=outside).
        mask: ``[B, N]`` boolean — ``True`` for annotated persons
            (``inout != -1``).
        weight: optional ``[B, N]`` per-person multiplicative weight applied
            on top of ``mask``. Used to down-weight pseudo-labelled persons.

    Returns:
        Tuple of ``(loss, {"inout_bce_loss": float})``.
    """
    if mask.sum() == 0:
        zero = torch.tensor(0.0, device=pred_logit.device)
        return zero, {"inout_bce_loss": 0.0}

    gt = gt_inout * mask
    bce_loss = F.binary_cross_entropy_with_logits(pred_logit, gt, reduction="none")
    if weight is None:
        loss = (bce_loss * mask).sum() / mask.sum()
    else:
        eff = mask.float() * weight.to(mask.device)
        loss = (bce_loss * eff).sum() / eff.sum().clamp(min=1e-6)
    return loss, {"inout_bce_loss": loss.item()}


# ---------------------------------------------------------------------------
# Social-gaze pair-head BCE
# ---------------------------------------------------------------------------


def _social_pair_bce(
    pred_logits: torch.Tensor,
    gt: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight: float,
) -> torch.Tensor:
    """Per-category BCE matching MTGS's ``train/losses.py:13-39``.

    Computes ``pos_weight``-adjusted BCE on every pair flagged by
    ``valid_mask``, then averages over the count of valid pairs.
    Returns a 0-d tensor (zero when no valid pairs in the batch).
    """
    if valid_mask.sum() == 0:
        return pred_logits.new_zeros(())
    # Zero-out -1 / pad entries before the BCE so they don't contribute
    # NaN to the per-element loss tensor.
    gt_clean = gt.float() * valid_mask.float()
    pos_w = pred_logits.new_tensor(pos_weight)
    per_elem = F.binary_cross_entropy_with_logits(
        pred_logits, gt_clean, pos_weight=pos_w, reduction="none",
    )
    masked = per_elem * valid_mask.float()
    return masked.sum() / (valid_mask.sum().float() + 1e-6)


def compute_social_pair_bce_loss(
    pred_logits: torch.Tensor,
    lah_gt: torch.Tensor,
    laeo_gt: torch.Tensor,
    coatt_gt: torch.Tensor,
    pair_valid_mask: torch.Tensor,
    pos_weight_lah: float = 3.0,
    pos_weight_laeo: float = 2.0,
    pos_weight_coatt: float = 2.0,
):
    """Per-category BCE for the social-pair head.

    Mirrors ``compute_social_loss`` in MTGS's ``train/losses.py`` (which uses
    ``social_loss`` per category and sums with equal coefficients):

      * Each category uses its own positive-class weight to compensate
        for class imbalance (LAH=3.0, LAEO=2.0, coatt=2.0 — defaults
        from the MTGS paper).
      * Per-element BCE, masked over pairs that are both valid (no
        padded person) and annotated (gt != -1) for that category, then
        normalised by the valid count.
      * The three category losses are summed with unit coefficients —
        the outer ``lambda_social`` (set in ``models_ge.py``) scales the
        total social term against ``lambda_lm`` / ``lambda_hm`` etc.

    Args:
        pred_logits: ``[B, P, 3]`` raw logits in channel order
            ``(lah, laeo, coatt)``. ``P = N*(N-1)`` for the
            collator-padded ``N``.
        lah_gt, laeo_gt, coatt_gt: ``[B, P]`` in ``{-1, 0, 1}``. ``-1``
            entries are treated as unannotated and masked out per
            category.
        pair_valid_mask: ``[B, P]`` boolean. ``True`` for pairs whose
            both ends are real persons (not padded slots). Should
            already be False for pairs touching a padded position.

    Returns:
        ``(total_loss, {"social_lah_loss", "social_laeo_loss",
        "social_coatt_loss", "social_total_loss"})``.
    """
    base_mask = pair_valid_mask.bool()

    lah_mask = base_mask & (lah_gt != -1)
    laeo_mask = base_mask & (laeo_gt != -1)
    coatt_mask = base_mask & (coatt_gt != -1)

    lah_loss = _social_pair_bce(
        pred_logits[..., 0], lah_gt, lah_mask, pos_weight_lah,
    )
    laeo_loss = _social_pair_bce(
        pred_logits[..., 1], laeo_gt, laeo_mask, pos_weight_laeo,
    )
    coatt_loss = _social_pair_bce(
        pred_logits[..., 2], coatt_gt, coatt_mask, pos_weight_coatt,
    )
    total = lah_loss + laeo_loss + coatt_loss
    logs = {
        "social_lah_loss": float(lah_loss.detach()),
        "social_laeo_loss": float(laeo_loss.detach()),
        "social_coatt_loss": float(coatt_loss.detach()),
        "social_total_loss": float(total.detach()),
    }
    return total, logs
