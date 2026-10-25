# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from typing import Dict, List
import itertools

import torch
import numpy as np

from omnigf.utils.utils import is_inside


def get_shuffle_idx(inout):
    """Shuffle indices corresponding to people"""
    inout = inout[1:]  # remove background person
    ann_people = np.where(inout != -1)[0]
    other_people = np.where(inout == -1)[0]
    ann_people = (ann_people[np.random.permutation(len(ann_people))] + 1).tolist()
    other_people = (other_people[np.random.permutation(len(other_people))] + 1).tolist()
    return [0] + other_people + ann_people


def get_lah_labels(pids, pairs_all, lah_labels_all):
    """Get lah labels for selected pids"""
    indices = torch.tensor(list(itertools.permutations(torch.arange(len(pids)), 2))).T
    pairs = [[pids[i].item(), pids[j].item()] for i, j in zip(indices[0], indices[1])]
    pairs = torch.tensor(pairs)
    lah_labels = torch.zeros(pairs.shape[0], dtype=torch.float32) - 1.0
    for idx, pair in enumerate(pairs):
        match_idx = torch.where(
            (pairs_all[:, 0] == pair[0]) * (pairs_all[:, 1] == pair[1])
        )[0]
        if match_idx.shape[0] > 0:
            lah_labels[idx] = lah_labels_all[match_idx[0]]
    return lah_labels


def get_social_gaze_predictions(
    prediction: Dict,
    image_width: int,
    image_height: int,
    num_people: int,
) -> List:
    """Get social gaze predictions"""

    # Prediction values
    head_bboxes = prediction["head_bboxes"]
    gaze_points = prediction["gaze_points"]
    lah = torch.zeros_like(prediction["lah"])
    laeo = torch.zeros_like(prediction["laeo"])
    coatt = prediction["coatt"]

    img_size = torch.tensor([image_width, image_height])

    pair_indices = torch.tensor(
        list(itertools.permutations(torch.arange(num_people + 1), 2))
    )

    # Iterate over pairs
    for ppid, pair in enumerate(pair_indices):
        # If padded person
        if pair[0] == 0 or pair[1] == 0:
            continue

        # Head bbox and gaze prediction for person 1
        head_bbox1 = head_bboxes[pair[0] - 1].tolist()
        gaze_pred1 = (gaze_points[pair[0] - 1] * img_size).tolist()

        # Head bbox and gaze prediction for person 2
        head_bbox2 = head_bboxes[pair[1] - 1].tolist()
        gaze_pred2 = (gaze_points[pair[1] - 1] * img_size).tolist()

        # Check whether person2 looks at person1
        if is_inside(head_bbox1, gaze_pred2):
            lah[0][ppid] = 1
            # Check whether person1 looks at person2
            if is_inside(head_bbox2, gaze_pred1):
                laeo[0][ppid] = 1

    # Peform arg max for lah
    lah_argmax = torch.zeros_like(lah)
    for i in range(num_people):
        valid_indices = torch.where(
            (pair_indices[:, 1] == i).int() * (pair_indices[:, 0] != 0).int()
        )[0]
        if valid_indices.shape[0] > 0:
            max_val, max_idx = torch.max(lah[0][valid_indices], 0)
            lah_argmax[0][valid_indices[max_idx]] = max_val

    # Peform arg max for laeo
    laeo_argmax = torch.zeros_like(laeo)
    for i in range(num_people):
        valid_indices = torch.where(
            (pair_indices[:, 1] == i).int() * (pair_indices[:, 0] != 0).int()
        )[0]
        if valid_indices.shape[0] > 0:
            max_val, max_idx = torch.max(laeo[0][valid_indices], 0)
            laeo_argmax[0][valid_indices[max_idx]] = max_val

    # Concatenate social gaze predictions
    social_preds = [lah_argmax, laeo_argmax, coatt]

    return social_preds
