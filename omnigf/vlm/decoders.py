# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""Heatmap decoders for VLM-based gaze prediction.

:class:`ConvTransposeHeatmapDecoder` produces ``[B, N, 64, 64]`` per-person
gaze heatmaps from one layer of Qwen VLM hidden states (~2 M params).
"""

from typing import List, Tuple, Union

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ConvTransposeHeatmapDecoder — Approach 3D variant A
# ---------------------------------------------------------------------------


class ConvTransposeHeatmapDecoder(nn.Module):
    """Lightweight heatmap decoder using ConvTranspose layers.

    Uses **last-layer** vision tokens conditioned on person hidden states.
    Grid size is inferred dynamically from the input.  ~2.1 M parameters.

    Attributes:
        NEEDS_MULTI_LAYER: ``False`` — only needs last LLM layer.
    """

    NEEDS_MULTI_LAYER = False

    def __init__(
        self,
        hidden_dim: int,
        proj_dim: int = 256,
        activation: str = "relu",
        negative_slope: float = 0.01,
    ):
        """
        Args:
            activation: ``"relu"`` (default, historical) or ``"leaky_relu"``.
                The ReLU before ``conv_out`` sits on a SINGLE channel, so once
                it saturates negative the whole heatmap freezes at
                ``conv_out.bias`` with no gradient to recover from. Measured on
                cached real features (600-1200 steps, GazeFollow, the training
                recipe): with ReLU that layer runs 55-99% dead at every model
                size and collapses outright on some seeds -- 3/3 Qwen3-VL-2B
                and 1/1 8B runs died at dist 0.7285 / AUC 0.5000, while 4B
                merely limped (85-90% dead). ``leaky_relu`` holds it at 0% dead
                on both 2B and 4B features and reaches a lower argmax error
                (0.026-0.061 vs 0.13-0.82).
                GELU is NOT a substitute: it still saturates (24-88% dead).
            negative_slope: slope for ``leaky_relu``. 0.01 and 0.2 both measured
                healthy.

        Note: ``activation`` changes the forward function, so a checkpoint must
        be evaluated with the same setting it was trained under. Default stays
        ``"relu"`` so existing checkpoints reproduce exactly.
        """
        super().__init__()
        self.proj_dim = proj_dim
        self.activation = activation

        self.person_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, proj_dim),
        )
        self.vision_proj = nn.Conv2d(hidden_dim, proj_dim, kernel_size=1)

        # Deconv: 14→29→59, then interpolate to 64×64.
        # No padding/output_padding, following attention-target-detection style.
        self.deconv1 = nn.ConvTranspose2d(proj_dim, 128, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 1, kernel_size=3, stride=2)
        self.conv_out = nn.Conv2d(1, 1, kernel_size=1)
        if activation == "leaky_relu":
            self.relu = nn.LeakyReLU(negative_slope, inplace=True)
        elif activation == "relu":
            self.relu = nn.ReLU(inplace=True)
        else:
            raise ValueError(
                f"decoder_activation must be 'relu' or 'leaky_relu', "
                f"got {activation!r}"
            )

        # conv_out sees a SINGLE post-ReLU channel, so its input is >= 0 and a
        # negative weight makes a positive peak unrepresentable: the cheapest
        # MSE descent direction is then to drive that lone ReLU to zero, which
        # also kills its own gradient. The decoder is stuck emitting
        # conv_out.bias at every pixel — a perfectly uniform heatmap whose
        # argmax is pixel 0, i.e. a constant predicted point.
        # PyTorch's default init here is U(-1, 1) (kaiming_uniform with
        # fan_in = 1), so the sign is a literal coin flip: a Qwen3-VL-2B run
        # drew -0.2977 and froze at dist 0.7285 / AUC 0.5000 for 7 epochs while
        # the 4B runs drew positive and trained normally.
        # Initialising conv_out as a positive identity removes the lottery.
        # This affects fresh runs ONLY — load_state_dict overwrites it, so
        # existing checkpoints are bit-for-bit unchanged.
        nn.init.ones_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(
        self,
        vision_hidden: torch.Tensor,
        person_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            vision_hidden: ``[B, V, D]`` — vision tokens (last LLM layer).
                ``V`` is the spatial token count (e.g. 196 for 14×14).
            person_hidden: ``[B, N, D]`` — person hidden states.

        Returns:
            ``[B, N, 64, 64]`` per-person gaze heatmaps.
        """
        B, V, D = vision_hidden.shape
        N = person_hidden.size(1)

        # Infer spatial grid from token count (assumes square grid)
        gh = gw = int(V ** 0.5)
        assert gh * gw == V, f"Vision tokens {V} is not a perfect square"

        vision_2d = vision_hidden.permute(0, 2, 1).view(B, D, gh, gw)
        vision_feat = self.vision_proj(vision_2d)       # [B, proj_dim, gh, gw]
        person_feat = self.person_proj(person_hidden)   # [B, N, proj_dim]

        # Person conditioning via element-wise multiply
        vision_expanded = vision_feat.unsqueeze(1).expand(-1, N, -1, -1, -1)
        person_expanded = person_feat.unsqueeze(-1).unsqueeze(-1)
        conditioned = vision_expanded * person_expanded  # [B, N, proj_dim, gh, gw]

        x = conditioned.view(B * N, self.proj_dim, gh, gw)
        x = self.relu(self.deconv1(x))   # 14→29
        x = self.relu(self.deconv2(x))   # 29→59
        x = self.conv_out(x)
        x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        return x.view(B, N, 64, 64)
