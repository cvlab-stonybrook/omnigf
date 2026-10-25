# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

import os
from typing import Tuple, Union, Dict, List
from enum import Enum, auto

import math
import einops
import torch

import json

import numpy as np

IGNORE_FILES = [".dircksum"]


def load_json_file(file: str) -> None:
    check_file(file)
    return json.load(open(file, "r"))


def save_json_file(data: Dict, file: str) -> None:
    folder = os.path.dirname(file)
    os.makedirs(folder, exist_ok=True)
    json.dump(data, open(file, "w"), indent=4)


def check_folder(path: str) -> None:
    assert os.path.isdir(path), f"Folder not found: {path}"


def check_file(path: str) -> None:
    assert os.path.isfile(path), f"File not found: {path}"


def list_folder(path: str) -> List:
    check_folder(path)
    files = sorted(os.listdir(path))
    return sorted([f for f in files if f not in IGNORE_FILES])


def get_experiment_name(experiment_folder: str) -> str:
    return "_".join(experiment_folder.split("/")[-2:])  # [day, time]


def pair(size):
    return size if isinstance(size, (list, tuple)) else (size, size)


class Stage(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    PREDICT = auto()


# TODO: duplicated
def expand_bbox(bboxes, img_w, img_h, k=0.1):
    """
    Expand bounding boxes by a factor of k.

    Args:
        bboxes: a tensor of size (B, 4) or (4,) containing B boxes or a single box in the format [xmin, ymin, xmax, ymax]
        k: a scalar value indicating the expansion factor
        img_w: a scalar value indicating the width of the image
        img_h: a scalar value indicating the height of the image

    Returns:
        A tensor of size (B, 4) or (4,) containing the expanded bounding boxes in the format [xmin, ymin, xmax, ymax].
    """
    if len(bboxes.shape) == 1:
        # Add batch dimension if only a single box is provided
        bboxes = bboxes.unsqueeze(0)

    # Compute the width and height of the bounding boxes
    bboxes_w = bboxes[:, 2] - bboxes[:, 0]
    bboxes_h = bboxes[:, 3] - bboxes[:, 1]

    # Compute expansion values
    expand_w = k * bboxes_w
    expand_h = k * bboxes_h

    # Expand the bounding boxes
    expanded_bboxes = torch.stack(
        [
            torch.clamp(bboxes[:, 0] - expand_w, min=0.0),
            torch.clamp(bboxes[:, 1] - expand_h, min=0.0),
            torch.clamp(bboxes[:, 2] + expand_w, max=img_w),
            torch.clamp(bboxes[:, 3] + expand_h, max=img_h),
        ],
        dim=1,
    )

    return expanded_bboxes.squeeze(0) if len(bboxes.shape) == 1 else expanded_bboxes


def expand_bbox_for_demo(bbox, img_w, img_h, k=0.1):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bbox[0] = max(0, bbox[0] - k * w)
    bbox[1] = max(0, bbox[1] - k * h)
    bbox[2] = min(img_w, bbox[2] + k * w)
    bbox[3] = min(img_h, bbox[3] + k * h)
    return bbox


def square_bbox(bboxes, img_width, img_height):
    """
    Adjust bounding boxes to be squared while ensuring the center of the box doesn't change.
    If the bounding box is too close to the edge, recenter the box to keep it within the image frame.

    Args:
        bboxes: a tensor of size (B, 4) containing B bounding boxes in the format [xmin, ymin, xmax, ymax]
        img_width: a scalar value indicating the width of the image
        img_height: a scalar value indicating the height of the image

    Returns:
        A tensor of size (B, 4) containing the squared bounding boxes.
    """
    n = len(bboxes)
    xmin = bboxes[:, 0]
    ymin = bboxes[:, 1]
    xmax = bboxes[:, 2]
    ymax = bboxes[:, 3]

    # Calculate original widths and heights
    widths = xmax - xmin
    heights = ymax - ymin

    # Calculate centers
    center_x = xmin + widths / 2
    center_y = ymin + heights / 2

    # Calculate maximum side length
    max_side_length = torch.max(widths, heights)

    # Calculate new xmin, ymin, xmax, ymax
    new_xmin = center_x - max_side_length / 2
    new_ymin = center_y - max_side_length / 2
    new_xmax = center_x + max_side_length / 2
    new_ymax = center_y + max_side_length / 2

    # Create the squared bounding boxes
    squared_bboxes = torch.stack([new_xmin, new_ymin, new_xmax, new_ymax], dim=1)

    return squared_bboxes


def gaussian_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    mx: Union[float, torch.Tensor] = 0.0,
    my: Union[float, torch.Tensor] = 0.0,
    sx: float = 1.0,
    sy: float = 1.0,
):
    out = (
        1
        / (2 * math.pi * sx * sy)
        * torch.exp(-((x - mx) ** 2 / (2 * sx**2) + (y - my) ** 2 / (2 * sy**2)))
    )
    return out


def generate_gaze_heatmap(
    gaze_pts: torch.Tensor, sigma: Union[int, Tuple] = 3, size: Union[int, Tuple] = 64
) -> torch.Tensor:
    """
    Function to generate a gaze heatmap from a set of gaze points.

    WARNING: the Gaussian is evaluated over the FULL grid and is **not**
    truncated — its float tail is non-zero over ~99% of a 64x64 map (4055/4096
    pixels), even though it carries only ~1% of the mass. Never treat
    ``heatmap > 0`` as "the gaze region": that mistake previously inflated the
    AUC positive mask ~10x and deflated single-annotation AUC to ~0.68 (see
    ``omnigf/performance/metrics.py::AUC``). Reference implementations
    (Chong et al. ``draw_labelmap``, gazelle ``vat_auc``) instead paint the
    Gaussian only inside a ``6*sigma + 1`` window, leaving the rest exactly 0,
    which is what makes ``> 0`` meaningful for them. Use a value threshold
    (e.g. ``> 0.01``) or build an explicit box from the gaze point instead.

    Args:
        gaze_pts (torch.Tensor): normalized gaze points (ie. [num_heads, gaze_x, gaze_y]) between [0, 1].
        sigma (Union[int, Tuple], optional): standard deviation. Defaults to 3.
        size (Union[int, Tuple], optional): spatial size of the output (ie. [width, height]). Defaults to 64.

    Returns:
        torch.Tensor: the gaze heatmap corresponding to gaze_pt
    """

    device = gaze_pts.device
    size = torch.tensor((size, size)) if isinstance(size, int) else torch.tensor(size)
    sigma = (
        torch.tensor((sigma, sigma)) if isinstance(sigma, int) else torch.tensor(sigma)
    )
    gaze_pts = gaze_pts * size
    num_heads = len(gaze_pts)

    heatmaps = torch.zeros(
        (num_heads, size[1], size[0]), dtype=torch.float, device=device
    )
    x = torch.arange(0, size[0], device=device)
    y = torch.arange(0, size[1], device=device)
    x, y = torch.meshgrid(x, y, indexing="xy")
    for hi, gp in enumerate(gaze_pts):
        if gp[1] > 0:
            heatmaps[hi] = gaussian_2d(x, y, gp[0], gp[1], sigma[0], sigma[1])
            heatmaps[hi] /= heatmaps[hi].max()

    return heatmaps


def generate_mask(bboxes, img_w, img_h):
    """
    Create a binary mask tensor where pixels inside the bounding boxes have a value of 1.

    Args:
        bboxes: a tensor of size (N, 4) or (4,) containing N or 1 bounding boxes in the format [xmin, ymin, xmax, ymax]
                normalized to [0, 1]
        img_w: a scalar value indicating the width of the image
        img_h: a scalar value indicating the height of the image

    Returns:
        A binary tensor of shape (N, 1, img_height, img_width) where pixels inside the bounding boxes
        have a value of 1.
    """

    ndim = bboxes.ndim
    if ndim == 1:
        bboxes = bboxes.unsqueeze(0)

    # Calculate pixel coordinates of bounding boxes
    xmin = (bboxes[:, 0] * img_w).long()
    ymin = (bboxes[:, 1] * img_h).long()
    xmax = (bboxes[:, 2] * img_w).long()
    ymax = (bboxes[:, 3] * img_h).long()

    # Determine the number of boxes
    num_boxes = bboxes.shape[0]

    # Create empty binary mask tensor
    mask = torch.zeros(
        (num_boxes, 1, img_h, img_w), dtype=torch.float32, device=bboxes.device
    )

    # Generate grid of indices
    grid_y, grid_x = torch.meshgrid(
        torch.arange(img_h, device=bboxes.device),
        torch.arange(img_w, device=bboxes.device),
        indexing="ij",
    )

    # Reshape grid indices for broadcasting
    grid_y = grid_y.view(1, img_h, img_w)
    grid_x = grid_x.view(1, img_h, img_w)

    # Determine if each pixel falls within any of the bounding boxes
    inside_mask = (
        (grid_x >= xmin.view(num_boxes, 1, 1))
        & (grid_x <= xmax.view(num_boxes, 1, 1))
        & (grid_y >= ymin.view(num_boxes, 1, 1))
        & (grid_y <= ymax.view(num_boxes, 1, 1))
    )

    # Set corresponding pixels to 1 in the mask tensor
    mask[inside_mask.unsqueeze(1)] = 1
    return mask.squeeze(0) if ndim == 1 else mask


def spatial_argmax2d(heatmap, normalize=True):
    """
    Function to locate the coordinates of the max value in the heatmap.
    Computation is done under no_grad() context.

    Args:
        heatmap (torch.Tensor): The input heatmap of shape (H, W) or (B, H, W).
        normalize (bool, optional): Specifies whether to normalize the argmax coordinates to [0, 1]. Defaults to True.

    Returns:
        torch.Tensor: The (normalized) argmax coordinates in the form (x, y) (i.e. shape (B, 2) or (2,))
    """

    with torch.no_grad():
        ndim = heatmap.ndim
        if ndim == 2:
            heatmap = heatmap.unsqueeze(0)

        points = (heatmap == torch.amax(heatmap, dim=(1, 2), keepdim=True)).nonzero()
        points = remove_duplicate_max(points)
        points = points[:, 1:].flip(1)  # (idx, y, x) -> (x, y)

        if normalize:
            points = points / torch.tensor(heatmap.size()[1:]).flip(0).to(
                heatmap.device
            )

        if ndim == 2:
            points = points[0]

    return points


def remove_duplicate_max(pts):
    """
    Function to remove duplicate rows based on the values of the first column (i.e. representing indices).
    The first occurence of each index value is kept.

    Args:
        pts (torch.Tensor): The points tensor of shape (N, 3) where 3 represents (index, y, x).

    Returns:
        torch.Tensor: Tensor of shape (M, 3) where M <= N after removing duplicates based on index value.
    """
    _, counts = torch.unique_consecutive(pts[:, 0], return_counts=True, dim=0)
    cum_sum = counts.cumsum(0)
    first_unique_idx = torch.cat((torch.tensor([0], device=pts.device), cum_sum[:-1]))
    return pts[first_unique_idx]


def build_2d_sincos_posemb(h, w, embed_dim=1024, temperature=10000.0):
    """Sine-cosine positional embeddings from MoCo-v3

    Source: https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    """
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")

    assert embed_dim % 4 == 0, (
        "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
    )

    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])

    pos_emb = torch.cat(
        [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
    )[None, :, :]
    pos_emb = einops.rearrange(pos_emb, "b (h w) d -> b d h w", h=h, w=w, d=embed_dim)

    return pos_emb


def generate_binary_gaze_heatmap(gaze_point, size=(64, 64)):
    """Draw the gaze point(s) on an empty canvas to produce a binary heatmap,
    where the location(s) of the gaze point(s) correspond to 1 while the rest
    is set to 0.

    Args:
        gaze_point (torch.Tensor): Gaze point(s) to draw.
        size (tuple, optional): Size of the output image [height, width]. Defaults to (64, 64).

    Returns:
        torch.Tensor: A binary gaze heatmap.
    """
    assert gaze_point.ndim <= 2, (
        f"Gaze point must be 1D or 2D, but found {gaze_point.ndim}D."
    )

    height, width = size
    # Pixel index convention: ``int(p * size)`` (floor) clamped to ``size - 1``,
    # matching the standard GazeFollow AUC protocol (gazelle
    # ``gazefollow_auc``). The ``* size`` scale also matches the rest of the
    # codebase — ``generate_gaze_heatmap`` centres the GT Gaussian at
    # ``p * size`` and ``spatial_argmax2d`` maps a pixel back via ``idx / size``
    # — whereas the previous ``p * (size - 1)`` was off by up to a full cell,
    # understating GazeFollow AUC by ~0.002 on average.
    # (Note: the Gaussian's argmax lands on ``round(p * size)``, so it can still
    # differ from this floor by one cell; that residual is inherent to the
    # reference protocol and is kept for parity.)
    gaze_point = (
        gaze_point * torch.tensor((width, height), device=gaze_point.device)
    ).int()
    gaze_point[..., 0] = gaze_point[..., 0].clamp(0, width - 1)
    gaze_point[..., 1] = gaze_point[..., 1].clamp(0, height - 1)
    binary_heatmap = torch.zeros(
        (height, width), device=gaze_point.device, dtype=torch.int
    )

    if gaze_point.ndim == 1:
        binary_heatmap[gaze_point[1], gaze_point[0]] = 1
    elif gaze_point.ndim == 2:  # gazefollow
        for gp in gaze_point:
            binary_heatmap[gp[1], gp[0]] = 1

    return binary_heatmap


def is_inside(head_bbox: np.ndarray, gaze_pt: np.ndarray) -> bool:
    """Return if gaze in inside head bounding box"""
    if (
        gaze_pt[0] > head_bbox[0]
        and gaze_pt[0] < head_bbox[2]
        and gaze_pt[1] > head_bbox[1]
        and gaze_pt[1] < head_bbox[3]
    ):
        return True
    return False
