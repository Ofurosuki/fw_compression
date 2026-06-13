"""EventTensorNet: shared event MLP + rank embedding + 2D spatial U-Net.

Input  ``events: [B, H, W, K, F]``  ->  logits ``[B, H, W, K, C]``.
Architecture follows ``initial_plan.md``; the only changes are (a) a configurable
input feature dim ``F`` (set per ablation feature_mode) and (b) automatic
reflect-padding so H, W need not be divisible by 4 (the U-Net pools twice).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F_


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=64):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base_channels * 2, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class EventTensorNet(nn.Module):
    def __init__(self, K, in_dim=5, emb_dim=32, num_classes=4, base_channels=64):
        super().__init__()
        self.K = K
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.event_mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.ReLU(inplace=True),
            nn.Linear(emb_dim, emb_dim), nn.ReLU(inplace=True),
        )
        self.rank_embedding = nn.Embedding(K, emb_dim)
        self.spatial_net = SmallUNet2D(K * emb_dim, K * num_classes, base_channels)

    def forward(self, events):
        """events: [B, H, W, K, F] -> logits [B, H, W, K, C]."""
        B, H, W, K, _ = events.shape
        assert K == self.K
        feat = self.event_mlp(events)                              # [B,H,W,K,D]
        rank = self.rank_embedding(torch.arange(K, device=events.device))
        feat = feat + rank.view(1, 1, 1, K, self.emb_dim)
        feat = feat.reshape(B, H, W, K * self.emb_dim).permute(0, 3, 1, 2).contiguous()

        # reflect-pad H,W up to a multiple of 4 (two 2x pools), crop logits back
        ph = (-H) % 4
        pw = (-W) % 4
        if ph or pw:
            feat = F_.pad(feat, (0, pw, 0, ph), mode="reflect")
        logits = self.spatial_net(feat)                            # [B,K*C,H',W']
        if ph or pw:
            logits = logits[:, :, :H, :W]
        logits = logits.permute(0, 2, 3, 1).contiguous().view(B, H, W, K, self.num_classes)
        return logits


def count_params(model):
    return sum(p.numel() for p in model.parameters())
