# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import torch
import torchmetrics as tm

from torchmetrics.functional.classification.auroc import binary_auroc

from omnigf.utils import (
    generate_binary_gaze_heatmap,
)


class Distance(tm.Metric):
    higher_is_better = False
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_observations", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_valid", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(
        self,
        gaze_point_pred: torch.Tensor,
        gaze_point_gt: torch.Tensor,
        inout_gt: torch.Tensor,
    ):
        mask = inout_gt == 1
        if mask.any():
            # Replace NaN predictions with center (0.5, 0.5) so that the
            # evaluation count stays consistent with baselines.
            pred = gaze_point_pred.clone()
            nan_mask = torch.isnan(pred[..., 0])
            pred[nan_mask] = 0.5
            self.sum_dist += (
                (gaze_point_gt[mask] - pred[mask]).pow(2).sum(1).sqrt().sum()
            )
            self.n_observations += mask.sum()
            self.n_valid += (mask & ~nan_mask).sum()

    def compute(self):
        if self.n_observations != 0:
            dist = self.sum_dist / self.n_observations  # type: ignore
        else:
            dist = torch.tensor(-1000.0, device=self.device)
        return dist


class GFTestDistance(tm.Metric):
    """GazeFollow multi-annotator distances, as reported in the paper.

    ``avg_dist``: distance from the prediction to the mean of the annotators'
    gaze points (the "Avg Dist." of Chong et al. and later work).
    ``min_dist``: distance from the prediction to the closest annotator's point.
    """

    higher_is_better = False
    full_state_update: bool = False

    def __init__(self):
        super().__init__()
        self.add_state("sum_avg_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_min_dist", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_observations", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "n_valid", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(self, gaze_point_pred: torch.Tensor, gaze_point_gt: torch.Tensor):
        for k, (gp_pred, gp_gt) in enumerate(zip(gaze_point_pred, gaze_point_gt)):
            # Replace NaN predictions with center (0.5, 0.5) so that the
            # evaluation count stays consistent with baselines.
            is_nan = torch.isnan(gp_pred).any()
            if is_nan:
                gp_pred = torch.tensor([0.5, 0.5], device=gp_pred.device)
            else:
                self.n_valid += 1
            gp_gt = gp_gt[gp_gt[:, 0] != -1]  # discard invalid gaze points

            # Distance from pred to the mean annotator point
            gp_gt_avg = gp_gt.mean(0)
            self.sum_avg_dist += (gp_gt_avg - gp_pred).pow(2).sum().sqrt()
            # Distance from pred to the closest annotator point
            self.sum_min_dist += (gp_gt - gp_pred).pow(2).sum(1).sqrt().min()
        self.n_observations += len(gaze_point_pred)

    def compute(self):
        if self.n_observations == 0:
            sentinel = torch.tensor(-1000.0, device=self.device)
            return sentinel, sentinel
        avg_dist = self.sum_avg_dist / self.n_observations
        min_dist = self.sum_min_dist / self.n_observations
        return avg_dist, min_dist


class AUC(tm.Metric):
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, tol_sigma: int = 3):
        """AUC for single-annotation datasets (VAT / ChildPlay / VideoCoAtt / …).

        Follows the standard VideoAttentionTarget protocol (Chong et al. /
        gazelle ``vat_auc``): the ground-truth positive region is a compact
        ``(6*tol_sigma + 1)``-side box centred on the gaze point (19×19 at 64×64
        for ``tol_sigma=3``), and the predicted heatmap is ranked against it.

        NOTE: the previous implementation binarised the GT with
        ``(gaze_heatmap_gt > 0)``. Because the Gaussian GT is not hard-truncated
        (its float tail is non-zero over ~99% of a 64×64 grid), that marked
        almost the whole frame as positive, so any well-localised (compact)
        prediction was scored as near-random — deflating AUC to ~0.68. The gaze
        point is recovered from the GT heatmap argmax (sub-pixel accurate for
        the single-Gaussian GT of these datasets), keeping the ``update``
        signature unchanged.
        """
        super().__init__()
        self.tol_sigma = int(tol_sigma)
        self.add_state("sum_auc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_observations", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def _box_target(self, hm_gt: torch.Tensor) -> torch.Tensor:
        """Binary tolerance box around the GT heatmap peak (the gaze point)."""
        H, W = hm_gt.shape[-2], hm_gt.shape[-1]
        idx = int(torch.argmax(hm_gt))
        gy, gx = idx // W, idx % W
        s = 3 * self.tol_sigma
        target = torch.zeros((H, W), dtype=torch.int, device=hm_gt.device)
        x1, y1 = max(0, gx - s), max(0, gy - s)
        x2, y2 = min(gx + s + 1, W - 1), min(gy + s + 1, H - 1)
        target[y1:y2, x1:x2] = 1
        return target

    def update(
        self,
        gaze_heatmap_pred: torch.Tensor,
        gaze_heatmap_gt: torch.Tensor,
        inout_gt: torch.Tensor,
    ):
        for hm_pred, hm_gt, io_gt in zip(gaze_heatmap_pred, gaze_heatmap_gt, inout_gt):
            if io_gt != 1:
                continue
            # Skip degenerate GT (no valid gaze target -> all-zero heatmap);
            # previously these silently contributed a 0/undefined AUC.
            if hm_gt.max() <= 0:
                continue
            target = self._box_target(hm_gt)
            self.sum_auc += binary_auroc(hm_pred.reshape(-1), target.reshape(-1))
            self.n_observations += 1

    def compute(self):
        if self.n_observations != 0:
            auc = self.sum_auc / self.n_observations
        else:
            auc = torch.tensor(-1000.0, device=self.device)
        return auc


class GFTestAUC(tm.Metric):
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, full_resolution: bool = False):
        """AUC for the GazeFollow test set (multi-annotator, single-pixel GT).

        The ground-truth map marks one pixel per valid annotator; AUC is
        computed per image and averaged.

        Args:
            full_resolution: When ``False`` (default) the metric scores on the
                model's native 64x64 output grid. When ``True`` it reproduces
                the reference protocol used by Chong et al.
                (``eval_on_gazefollow.py``) and Gaze-LLE / gazelle
                (``gazefollow_auc``): the predicted heatmap is bilinearly
                upsampled to the ORIGINAL image size and the GT pixels are
                placed at that resolution. Requires ``img_size`` in
                :meth:`update`; falls back to the 64x64 path without it.

        The two settings agree to within ~0.001 in simulation (AUC is a rank
        metric and bilinear upsampling preserves the heatmap's level sets), but
        the full-resolution path removes the protocol difference entirely when
        comparing against published numbers.
        """

        super().__init__()
        self.full_resolution = bool(full_resolution)
        self.add_state("sum_auc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_observations", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

    def update(
        self,
        gaze_heatmap_pred: torch.Tensor,
        gaze_pt: torch.Tensor,
        img_size: torch.Tensor = None,
    ):
        """
        Args:
            gaze_heatmap_pred: ``[B, h, w]`` predicted heatmaps.
            gaze_pt: ``[B, A, 2]`` annotator points, invalid rows marked ``-1``.
            img_size: ``[B, 2]`` original ``(width, height)`` per sample. Only
                used when ``full_resolution=True``.
        """
        size = gaze_heatmap_pred.shape[1:]  # (b, h, w) >> (h, w)
        use_full = self.full_resolution and img_size is not None
        for i, (hm_pred, gp_gt) in enumerate(zip(gaze_heatmap_pred, gaze_pt)):
            # ``>= 0`` matches the reference filter (Chong ``multi_hot_targets``);
            # equivalent to the historical ``!= -1`` since padding is exactly -1.
            gp_gt = gp_gt[gp_gt[:, 0] >= 0]
            if gp_gt.numel() == 0:
                continue
            if use_full:
                w, h = int(img_size[i][0]), int(img_size[i][1])
                pred = torch.nn.functional.interpolate(
                    hm_pred[None, None].float(), size=(h, w),
                    mode="bilinear", align_corners=False,
                ).squeeze(0).squeeze(0)
                hm_gt_binary = generate_binary_gaze_heatmap(gp_gt, size=(h, w))
            else:
                pred = hm_pred
                hm_gt_binary = generate_binary_gaze_heatmap(gp_gt, size=size)
            self.sum_auc += binary_auroc(
                pred.reshape(-1).float(), hm_gt_binary.reshape(-1),
            )
            self.n_observations += 1

    def compute(self):
        auc = self.sum_auc / self.n_observations
        return auc
