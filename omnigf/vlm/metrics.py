# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""Evaluation metrics for VLM gaze prediction."""

import torch
import torchmetrics as tm


class MultiAccuracy(tm.Metric):
    """Multi-label top-k accuracy following SemGaze.

    A prediction is correct if any of the top-k predicted classes matches
    any of the valid target classes (multiple GT labels per sample).
    """

    higher_is_better = True
    full_state_update: bool = False

    def __init__(self, top_k: int = 1, ignore_index: int = -1):
        super().__init__()
        self.top_k = top_k
        self.ignore_index = ignore_index
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args:
            preds: ``[M, num_classes]`` logits.
            target: ``[M, K]`` multi-label target IDs (padded with ignore_index).
        """
        mask = (target != self.ignore_index).any(dim=1)
        preds = preds[mask]
        target = target[mask]
        if preds.shape[0] == 0:
            return

        top_k_preds = preds.topk(self.top_k, dim=1)[1]  # [M, top_k]
        # Check if any top-k prediction matches any target label
        target_expanded = target.unsqueeze(1)  # [M, 1, K]
        correct = torch.any(
            top_k_preds.unsqueeze(2) == target_expanded, dim=1,
        ).any(dim=1).sum()
        self.correct += correct
        self.total += preds.size(0)

    def compute(self):
        if self.total == 0:
            return torch.tensor(0.0)
        return self.correct.float() / self.total
