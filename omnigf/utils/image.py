# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

from typing import Tuple, List

import cv2
import numpy as np

# Image normalization
IMG_MEAN = [0.44232, 0.40506, 0.36457]
IMG_STD = [0.28674, 0.27776, 0.27995]


def draw_rectangle(
    image: np.ndarray,
    x1y1: Tuple[int, int],
    x2y2: Tuple[int, int],
    color: List[int] = [0, 0, 0],
    thickness: int = 1,
) -> None:
    cv2.rectangle(image, x1y1, x2y2, color, thickness)


def draw_circle(
    image: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    color: List[int] = [0, 0, 0],
    thickness: int = 1,
) -> None:
    cv2.circle(image, center, radius, color, thickness)


def draw_line(
    image: np.ndarray,
    point1: Tuple[int, int],
    point2: Tuple[int, int],
    color: List[int] = [0, 0, 0],
    thickness: int = 1,
) -> None:
    cv2.line(image, point1, point2, color, thickness)


def draw_arrowed_line(
    image: np.ndarray,
    point1: Tuple[int, int],
    point2: Tuple[int, int],
    color: List[int] = [0, 0, 0],
    thickness: int = 1,
) -> None:
    cv2.arrowedLine(image, point1, point2, color, thickness)


def draw_text(
    image: np.ndarray,
    text: str,
    location: Tuple[int, int],
    scale: float,
    color: List[int] = [255, 255, 255],
    thickness: int = 1,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    style = cv2.LINE_AA
    cv2.putText(image, text, location, font, scale, color, thickness, style)


def get_text_size(
    text: str,
    scale: float,
    thickness: int = 1,
) -> Tuple[int, int]:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    return (text_w, text_h)
