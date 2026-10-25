# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Qiaomu Miao
# SPDX-License-Identifier: GPL-3.0-only

"""Prediction heads for VLM-based gaze prediction.

These lightweight modules are attached to the VLM backbone's hidden states
to produce structured spatial predictions.
"""

import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InOutHead(nn.Module):
    """MLP predicting a single in/out logit per person.

    Used by **Approach 3C** where in/out is handled by a separate BCE head
    rather than through the patch distribution.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, person_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            person_hidden: ``[B, N, hidden_dim]``.

        Returns:
            ``[B, N]`` — in/out logits (before sigmoid).
        """
        return self.mlp(person_hidden).squeeze(-1)


class SocialPairHead(nn.Module):
    """MLP predicting per-pair social-gaze logits.

    For every directed pair ``(i, j)`` in
    ``itertools.permutations(range(N), 2)``, computes 3 logits — one for
    each of the social-gaze categories: LAH (Looking At Head, directed),
    LAEO (Looking At Each Other, symmetric), and coatt (Co-attention /
    Shared Attention, symmetric).

    The pair feature is the simple concatenation
    ``[h_i, h_j]``: relies on the LLM's already-fully-attended hidden
    states to encode the cross-person relations. The MLP just decodes
    the pair-wise relation; no further interaction modeling needed.

    Pair-ordering convention follows the MTGS source datasets and
    :func:`itertools.permutations` — pair index ``k`` corresponds to
    ordered pair ``(i, j) = list(permutations(range(N), 2))[k]``. Per
    MTGS convention "pair [a, b] → b looks at a's head", position 0 is
    the LAH **target** and position 1 is the LAH **source/gazer**.
    """

    def __init__(
        self,
        hidden_dim: int,
        mid_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 3),  # [lah, laeo, coatt]
        )

    @staticmethod
    def _pair_indices(n: int, device) -> torch.Tensor:
        """Return ``[N*(N-1), 2]`` of directed permutation pairs."""
        return torch.tensor(
            list(itertools.permutations(range(n), 2)),
            device=device, dtype=torch.long,
        )

    def forward(self, person_hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            person_hidden: ``[B, N, hidden_dim]``.

        Returns:
            ``[B, N*(N-1), 3]`` — per-pair logits before sigmoid for
            channels ``[lah, laeo, coatt]``.
        """
        B, N, D = person_hidden.shape
        if N < 2:
            return person_hidden.new_zeros(B, 0, 3)
        idx = self._pair_indices(N, person_hidden.device)  # [P, 2]
        # Gather both ends of every directed pair.
        h_i = person_hidden[:, idx[:, 0], :]  # [B, P, D]
        h_j = person_hidden[:, idx[:, 1], :]  # [B, P, D]
        pair_feat = torch.cat([h_i, h_j], dim=-1)  # [B, P, 2D]
        logits = self.mlp(pair_feat)  # [B, P, 3]
        return logits
