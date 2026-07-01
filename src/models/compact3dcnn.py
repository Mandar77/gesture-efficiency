"""Compact 3D-CNN — pipeline-validation baseline (BRIEF M3) and smoke model.

A small, from-scratch 3D convolutional network for video-clip classification.
With `width=8, depth=2` it is tiny enough for the 1-2 minute smoke test; with
default width it is a credible from-scratch baseline on Jester to validate the
data + train + bench pipeline before the foundation-model track.

Input:  [B, C, T, H, W]   (channels-first, time-major)
Output: [B, num_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.utils.registry import register


class _ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride=(1, 2, 2)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


@register("model", "compact3dcnn")
class Compact3DCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 27,
        in_channels: int = 3,
        width: int = 32,
        depth: int = 4,
        dropout: float = 0.2,
        **_ignore,
    ):
        super().__init__()
        chans = [in_channels] + [width * (2**i) for i in range(depth)]
        blocks = []
        for i in range(depth):
            # Downsample time only every other block so short clips survive.
            t_stride = 2 if i % 2 == 1 else 1
            blocks.append(_ConvBlock(chans[i], chans[i + 1], stride=(t_stride, 2, 2)))
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(chans[-1], num_classes)
        self.feature_dim = chans[-1]  # exposed for feature-distillation projection

    def forward(self, x, return_features: bool = False):
        f = self.features(x)
        f = self.pool(f).flatten(1)
        logits = self.classifier(self.dropout(f))
        if return_features:
            return logits, f
        return logits


@register("model", "dummy")
class DummyClassifier(nn.Module):
    """Absolutely minimal model for the fastest possible smoke pass."""

    def __init__(self, num_classes: int = 27, in_channels: int = 3, **_ignore):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        self.feature_dim = in_channels

    def forward(self, x, return_features: bool = False):
        f = self.pool(x).flatten(1)
        logits = self.fc(f)
        if return_features:
            return logits, f
        return logits
