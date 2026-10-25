# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""MTGS-style social-gaze pair metrics for the VLM heads-based path.

Ports MTGS's ``performance/compute_metrics.py:88-215`` so that VLM runs
under ``vlm.use_social_pair_head=True`` produce numbers directly
comparable to the MTGS paper:

* **LAH** — per-(image, gazer) binary observation. Each gazer is a
  candidate iff ``inout_gt == 1`` AND has at least one annotated
  outgoing LAH pair. The prediction is the per-gazer argmax score; the
  GT label is binary "is the gazer annotated as looking at any head".
* **LAEO** — per-(image, annotated pair). The prediction is sparsified
  to the per-gazer top-1 partner (every other pair gets score 0 →
  classified negative at threshold 0.5).
* **coatt** — per-(image, annotated pair). No argmax filter; raw
  per-pair predictions.

For each category we report the same five numbers as MTGS at threshold
0.5: ``ap``, ``auc``, ``precision``, ``recall``, ``f1``. AP and AUC are
threshold-free; P/R/F1 use ``score > thr``.
"""

from __future__ import annotations

import itertools
import logging
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

CATEGORIES = ("LAH", "LAEO", "coatt")


# ---------------------------------------------------------------------------
# Per-batch -> flat arrays
# ---------------------------------------------------------------------------


def _pair_indices_table(n: int) -> torch.Tensor:
    """Cached-friendly ``[N*(N-1), 2]`` long tensor of permutation pairs."""
    return torch.tensor(
        list(itertools.permutations(range(n), 2)), dtype=torch.long,
    )


def _per_gazer_lah(
    lah_pred: torch.Tensor,   # [N*(N-1)] sigmoid probs
    lah_gt: torch.Tensor,     # [N*(N-1)] in {-1, 0, 1}
    inout_gt: torch.Tensor,   # [N] in {-1, 0, 1}
    pair_indices: torch.Tensor,  # [N*(N-1), 2]
    n: int,
    force_io_one: bool = False,
):
    """One sample's per-gazer LAH observations, MTGS-style.

    Mirrors ``compute_metrics.py:88-126``. The gazer is at position 1 of
    the pair tuple (per MTGS convention "pair [a, b] = b looks at a").

    ``force_io_one``: when True, every position is treated as a candidate
    gazer (skips the ``inout==1`` filter). This matches MTGS's behavior
    on GazeFollow (``compute_metrics.py:96-97`` sets ``io = 1`` for GF).
    The inner ``(lah_gt != -1).sum() == 0`` annotation filter still
    applies, so positions without any annotated outgoing LAH pair are
    skipped regardless.

    Returns (gt_list, pred_list) both Python lists of floats — to be
    extended by the caller across the test set, then aggregated.
    """
    gts: List[float] = []
    preds: List[float] = []
    for pi in range(n):
        if not force_io_one and int(inout_gt[pi].item()) != 1:
            continue
        # Pairs where pi is at position 1 (gazer).
        valid_idx = (pair_indices[:, 1] == pi).nonzero(as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        # Skip the gazer if all outgoing LAH pairs are unannotated.
        if (lah_gt[valid_idx] != -1).sum() == 0:
            continue
        # Argmax predicted score among valid outgoing pairs.
        sub = lah_pred[valid_idx]
        max_val, max_off = torch.max(sub, dim=0)
        # GT label: "is the gazer annotated as looking at anyone?"
        labels = lah_gt[valid_idx]
        ann = labels[labels != -1]
        gt_label = int(min(int(ann.sum().item()), 1))
        # Predicted score: mirrors MTGS exactly
        # (``compute_metrics.py:107-122``):
        #
        #   lah_pred_argmax[valid][max_off] = max_val   # zero elsewhere
        #   if any positive label exists:
        #       gt_idx = first positive offset
        #       score = lah_pred_argmax[valid][gt_idx]
        #             = max_val   if max_off == gt_idx
        #             = 0.0       otherwise
        #   else:
        #       score = max_val
        pos_idx = (labels == 1).nonzero(as_tuple=False).flatten()
        if pos_idx.numel() > 0:
            gt_first = int(pos_idx[0].item())
            score = float(max_val.item()) if gt_first == int(max_off.item()) else 0.0
        else:
            score = float(max_val.item())
        preds.append(score)
        gts.append(gt_label)
    return gts, preds


def _per_gazer_argmax_laeo(
    laeo_pred: torch.Tensor,   # [N*(N-1)]
    pair_indices: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Sparsify ``laeo_pred`` so only the per-gazer top-1 pair carries
    its score; every other pair is zeroed. Mirrors
    ``compute_metrics.py:131-142``. The "exclude pos==0" filter from
    MTGS isn't applied here because our relabelled space starts from 0
    (no dummy slot at position 0)."""
    sparse = torch.zeros_like(laeo_pred)
    for pi in range(n):
        valid_idx = (pair_indices[:, 1] == pi).nonzero(as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        max_val, max_off = torch.max(laeo_pred[valid_idx], dim=0)
        sparse[valid_idx[max_off]] = max_val
    return sparse


# ---------------------------------------------------------------------------
# Aggregation entry point
# ---------------------------------------------------------------------------


def aggregate_pair_metrics(
    records: List[Dict],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute MTGS-style per-category AP/AUC/P/R/F1 across the test set.

    Each record is::

        {
            "lah_pred":   [P]  sigmoid probs in [0, 1],
            "laeo_pred":  [P],
            "coatt_pred": [P],
            "lah_gt":     [P]  in {-1, 0, 1},
            "laeo_gt":    [P],
            "coatt_gt":   [P],
            "inout_gt":   [N]  in {-1, 0, 1},
            "pair_valid_mask": [P] bool — True for pairs whose endpoints
                are both real persons (not padded slots),
            "n_people":   int  (= N),
        }

    Tensors may be on CPU; the function operates entirely in numpy after
    a per-record extraction phase.

    Returns a flat dict with keys
    ``LAH/{ap,auc,precision,recall,f1}`` etc., plus diagnostic
    ``LAH/n_obs`` etc.
    """
    try:
        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
            precision_score,
            recall_score,
            f1_score,
        )
    except ImportError:
        logger.error("scikit-learn not available; pair-head metrics skipped.")
        return {}

    # Per-gazer LAH
    lah_gt_all: List[float] = []
    lah_pred_all: List[float] = []
    # Per-pair LAEO (with per-gazer argmax filtering on pred)
    laeo_gt_all: List[float] = []
    laeo_pred_all: List[float] = []
    # Per-pair coatt (no argmax)
    coatt_gt_all: List[float] = []
    coatt_pred_all: List[float] = []

    for rec in records:
        n = int(rec["n_people"])
        if n < 2:
            continue
        pair_indices = _pair_indices_table(n)
        valid_mask = rec["pair_valid_mask"].cpu().bool()
        # Slice everything to the real-pair subset (drops padding-pair
        # rows so the per-image permutation grid matches ``range(n)``).
        # The collator builds the padded grid via permutations(max_V),
        # so we re-extract the in-V subset using ``valid_mask``.
        lah_pred = rec["lah_pred"].cpu()[valid_mask]
        laeo_pred = rec["laeo_pred"].cpu()[valid_mask]
        coatt_pred = rec["coatt_pred"].cpu()[valid_mask]
        lah_gt = rec["lah_gt"].cpu()[valid_mask]
        laeo_gt = rec["laeo_gt"].cpu()[valid_mask]
        coatt_gt = rec["coatt_gt"].cpu()[valid_mask]
        inout_gt = rec["inout_gt"].cpu()
        # Sanity: extracted tensors should be of length n*(n-1)
        if lah_pred.numel() != n * (n - 1):
            logger.warning(
                "pair_valid_mask count %d != n*(n-1) = %d for n=%d; skipping",
                lah_pred.numel(), n * (n - 1), n,
            )
            continue

        # ---- LAH per-gazer ----
        gts, preds = _per_gazer_lah(
            lah_pred, lah_gt, inout_gt[:n], pair_indices, n,
            force_io_one=bool(rec.get("is_gazefollow", False)),
        )
        lah_gt_all.extend(gts)
        lah_pred_all.extend(preds)

        # ---- LAEO: per-pair with per-gazer argmax filter ----
        laeo_pred_sparse = _per_gazer_argmax_laeo(laeo_pred, pair_indices, n)
        laeo_mask = (laeo_gt != -1)
        if laeo_mask.any():
            laeo_gt_all.extend(laeo_gt[laeo_mask].tolist())
            laeo_pred_all.extend(laeo_pred_sparse[laeo_mask].tolist())

        # ---- coatt: per-pair, no argmax ----
        coatt_mask = (coatt_gt != -1)
        if coatt_mask.any():
            coatt_gt_all.extend(coatt_gt[coatt_mask].tolist())
            coatt_pred_all.extend(coatt_pred[coatt_mask].tolist())

    report: Dict[str, float] = {}

    def _emit(cat: str, gt: List[float], pred: List[float]):
        if not gt:
            return
        gt_np = np.asarray(gt, dtype=np.int64)
        pred_np = np.asarray(pred, dtype=np.float64)
        n_obs = int(gt_np.shape[0])
        n_pos = int(gt_np.sum())
        report[f"{cat}/n_obs"] = float(n_obs)
        report[f"{cat}/n_pos"] = float(n_pos)
        # AP / AUC need both classes present.
        if 0 < n_pos < n_obs:
            try:
                report[f"{cat}/ap"] = float(average_precision_score(gt_np, pred_np))
                report[f"{cat}/auc"] = float(roc_auc_score(gt_np, pred_np))
            except ValueError:
                pass
        # P/R/F1 at threshold.
        pred_thr = (pred_np > threshold).astype(np.int64)
        try:
            report[f"{cat}/precision"] = float(
                precision_score(gt_np, pred_thr, zero_division=0)
            )
            report[f"{cat}/recall"] = float(
                recall_score(gt_np, pred_thr, zero_division=0)
            )
            report[f"{cat}/f1"] = float(
                f1_score(gt_np, pred_thr, zero_division=0)
            )
        except ValueError:
            pass

    _emit("LAH", lah_gt_all, lah_pred_all)
    _emit("LAEO", laeo_gt_all, laeo_pred_all)
    _emit("coatt", coatt_gt_all, coatt_pred_all)
    return report


__all__ = ["aggregate_pair_metrics", "CATEGORIES"]
