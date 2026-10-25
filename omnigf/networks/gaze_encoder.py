# coding=utf-8

# SPDX-FileCopyrightText: Copyright 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Anshul Gupta <anshul.gupta@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Head-crop gaze encoder, taken from the MTGS network definition."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GazeEncoder(nn.Module):
    def __init__(self, token_dim=768, feature_dim=512, arch="resnet18"):
        """Per-person head-crop encoder.

        Args:
            arch: backbone architecture, ``"resnet18"`` (default, matches the
                Gaze360 checkpoint) or ``"resnet50"`` (needed to accept the
                EyeDiap-pretrained trunk from Chong et al.'s
                ``attention-target-detection`` face pathway, which is a
                standard ResNet-50 for ``conv1``/``layer1``-``layer4``).
                ``embed_dim`` below is derived from a dummy forward, so the
                projection heads resize themselves: 512 for r18, 2048 for r50.
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.token_dim = token_dim
        self.arch = arch

        if arch == "resnet18":
            base = models.resnet18(weights=None)  # type: ignore
        elif arch == "resnet50":
            base = models.resnet50(weights=None)  # type: ignore
        else:
            raise ValueError(f"GazeEncoder arch={arch!r} invalid")
        self.backbone = nn.Sequential(*list(base.children())[:-1])

        dummy_head = torch.empty((1, 3, 224, 224))
        dummy_head = self.backbone(dummy_head)
        embed_dim = dummy_head.size(1)

        self.gaze_proj = nn.Sequential(
            nn.Linear(embed_dim, token_dim),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )
        self.pos_proj = nn.Linear(4, token_dim)

        self.gaze_predictor = nn.Sequential(  # self.gaze_predictor
            nn.Linear(embed_dim, feature_dim),
            nn.ReLU(inplace=True),
            # 2 = number of outputs (x, y) unit vector
            nn.Linear(feature_dim, 2),
            nn.Tanh(),
        )

        # Initialize weights
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_backbone(self, head):
        b, n, c, h, w = head.shape

        gaze_emb = self.backbone(head.view(-1, c, h, w)).flatten(
            1, -1
        )  # (b*n, embed_dim)

        return gaze_emb

    def forward_head(self, gaze_emb, head_bbox):
        b, n, p = head_bbox.shape
        head_bbox_emb = self.pos_proj(head_bbox.view(-1, p))  # (b*n, token_dim)

        gaze_token = (
            self.gaze_proj(gaze_emb.view(b * n, -1)) + head_bbox_emb
        )  # (b*n, token_dim)
        gaze_token = gaze_token.view(b, n, -1)  # (b, n, token_dim)

        gaze_vec = self.gaze_predictor(gaze_emb.view(b * n, -1))  # (b*n, 2)
        # normalize to unit vector
        gaze_vec = F.normalize(gaze_vec, p=2, dim=1)
        gaze_vec = gaze_vec.view(b, n, -1)  # (b, n, 2)

        return gaze_token, gaze_vec

