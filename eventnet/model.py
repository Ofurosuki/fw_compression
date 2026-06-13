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

    def forward(self, events, valid=None):
        """events: [B, H, W, K, F] -> logits [B, H, W, K, C]. ``valid`` unused (V1)."""
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


# --------------------------------------------------------------------------- #
# V2: cross-event attention (ray-relational features) + deeper U-Net.
# --------------------------------------------------------------------------- #
class CrossEventAttention(nn.Module):
    """Per-pixel self-attention over the K events of one ray. Lets each return
    attend to the others, learning relational cues (brightness rank, "is there a
    return behind me", inter-echo gaps) in a data-driven, permutation-aware way —
    the robust alternative to hand-crafted scalars like behind_energy that turned
    out depth-confounded. Padded (invalid) events are masked out."""

    def __init__(self, dim, heads=4, layers=2, ff_mult=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * ff_mult,
                batch_first=True, dropout=0.0, activation="gelu")
            for _ in range(layers)])

    def forward(self, x, valid):
        B, H, W, K, D = x.shape
        xs = x.reshape(B * H * W, K, D)
        mask = ~valid.reshape(B * H * W, K)              # True = pad (ignore)
        allpad = mask.all(dim=1)                          # background pixels
        mask = mask.clone()
        mask[allpad, 0] = False                           # unmask one slot -> no NaN softmax
        for layer in self.layers:
            xs = layer(xs, src_key_padding_mask=mask)
        return xs.reshape(B, H, W, K, D)


class UNet2D(nn.Module):
    """U-Net with a configurable number of 2x downsampling levels."""

    def __init__(self, in_channels, out_channels, base_channels=64, levels=3):
        super().__init__()
        self.levels = levels
        self.encs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        chans = []
        for l in range(levels):
            oc = base_channels * (2 ** l)
            self.encs.append(ConvBlock(ch, oc))
            self.pools.append(nn.MaxPool2d(2))
            chans.append(oc)
            ch = oc
        self.bottleneck = ConvBlock(ch, base_channels * (2 ** levels))
        self.ups = nn.ModuleList()
        self.decs = nn.ModuleList()
        bc = base_channels * (2 ** levels)
        for l in reversed(range(levels)):
            oc = chans[l]
            self.ups.append(nn.ConvTranspose2d(bc, oc, 2, stride=2))
            self.decs.append(ConvBlock(oc * 2, oc))
            bc = oc
        self.out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x):
        skips = []
        h = x
        for enc, pool in zip(self.encs, self.pools):
            h = enc(h)
            skips.append(h)
            h = pool(h)
        h = self.bottleneck(h)
        for up, dec, skip in zip(self.ups, self.decs, reversed(skips)):
            h = dec(torch.cat([up(h), skip], dim=1))
        return self.out(h)


class EventTensorNetV2(nn.Module):
    """Event MLP + rank embedding -> cross-event attention -> deeper 2D U-Net."""

    def __init__(self, K, in_dim=5, emb_dim=48, num_classes=4, base_channels=64,
                 attn_heads=4, attn_layers=2, unet_levels=3):
        super().__init__()
        self.K = K
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.unet_levels = unet_levels
        self.event_mlp = nn.Sequential(
            nn.Linear(in_dim, emb_dim), nn.GELU(),
            nn.Linear(emb_dim, emb_dim), nn.GELU(),
        )
        self.rank_embedding = nn.Embedding(K, emb_dim)
        self.cross_event = CrossEventAttention(emb_dim, attn_heads, attn_layers)
        self.spatial_net = UNet2D(K * emb_dim, K * num_classes, base_channels, levels=unet_levels)

    def forward(self, events, valid):
        """events: [B,H,W,K,F], valid: [B,H,W,K] bool -> logits [B,H,W,K,C]."""
        B, H, W, K, _ = events.shape
        assert K == self.K
        feat = self.event_mlp(events)                              # [B,H,W,K,D]
        rank = self.rank_embedding(torch.arange(K, device=events.device))
        feat = feat + rank.view(1, 1, 1, K, self.emb_dim)
        feat = self.cross_event(feat, valid)                       # [B,H,W,K,D]
        feat = feat.reshape(B, H, W, K * self.emb_dim).permute(0, 3, 1, 2).contiguous()

        m = 2 ** self.unet_levels
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            feat = F_.pad(feat, (0, pw, 0, ph), mode="reflect")
        logits = self.spatial_net(feat)
        if ph or pw:
            logits = logits[:, :, :H, :W]
        logits = logits.permute(0, 2, 3, 1).contiguous().view(B, H, W, K, self.num_classes)
        return logits


def build_model(arch, K, in_dim, num_classes=4, emb_dim=None, base_channels=64,
                attn_heads=4, attn_layers=2, unet_levels=3):
    if arch == "v1":
        return EventTensorNet(K, in_dim=in_dim, emb_dim=emb_dim or 32,
                              num_classes=num_classes, base_channels=base_channels)
    if arch == "v2":
        return EventTensorNetV2(K, in_dim=in_dim, emb_dim=emb_dim or 48,
                                num_classes=num_classes, base_channels=base_channels,
                                attn_heads=attn_heads, attn_layers=attn_layers,
                                unet_levels=unet_levels)
    raise ValueError(arch)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
