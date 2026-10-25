# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""VLM-based gaze prediction module.

This package implements a VLM fine-tuning pipeline for multi-person gaze
target prediction using Qwen2.5-VL / Qwen3-VL.  Two approach variants are
supported (see :class:`~omnigf.vlm.models_ge.VLMGazeModelGE`).
"""

from omnigf.vlm.collator import VLMGazeCollator
from omnigf.vlm.dataset import VLMGazeDataModule, VLMGazeDataset
from omnigf.vlm.decoders import (
    ConvTransposeHeatmapDecoder,
)
from omnigf.vlm.evaluation import extract_predictions
from omnigf.vlm.heads import (
    InOutHead,
)
from omnigf.vlm.losses import (
    compute_heatmap_mse_loss,
    compute_inout_bce_loss,
)
from omnigf.vlm.models_ge import VLMGazeModelGE
from omnigf.vlm.parser import MultiPersonOutputParser

__all__ = [
    # Models
    "VLMGazeModelGE",
    # Dataset / DataModule
    "VLMGazeDataset",
    "VLMGazeDataModule",
    # Collator
    "VLMGazeCollator",
    # Heads
    "InOutHead",
    # Decoders
    "ConvTransposeHeatmapDecoder",
    # Losses
    "compute_heatmap_mse_loss",
    "compute_inout_bce_loss",
    # Evaluation
    "extract_predictions",
    # Parser
    "MultiPersonOutputParser",
]
