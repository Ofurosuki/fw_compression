"""Off-sensor DNN decoders mapping latent ``z [B, K]`` to ``x_hat [B, T]``.

The decoder is allowed to be heavy (it runs off-sensor). We start with the MLP
specified in the research plan and provide a slightly deeper variant for later.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """Linear(K,256) -> ReLU -> Linear(256,512) -> ReLU -> Linear(512,T)."""

    def __init__(self, K: int, T: int, hidden=(256, 512), nonneg: bool = True):
        super().__init__()
        dims = [K, *hidden, T]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self.nonneg = nonneg

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        if self.nonneg:
            # waveforms are non-negative photon counts
            x = torch.relu(x)
        return x


class DeepMLPDecoder(nn.Module):
    """Deeper MLP with residual blocks; optional for later experiments."""

    def __init__(self, K: int, T: int, width: int = 512, depth: int = 4, nonneg: bool = True):
        super().__init__()
        self.inp = nn.Linear(K, width)
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Linear(width, width), nn.ReLU(inplace=True), nn.Linear(width, width)) for _ in range(depth)]
        )
        self.out = nn.Linear(width, T)
        self.act = nn.ReLU(inplace=True)
        self.nonneg = nonneg

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.act(self.inp(z))
        for blk in self.blocks:
            h = self.act(h + blk(h))
        x = self.out(h)
        if self.nonneg:
            x = torch.relu(x)
        return x


DECODER_REGISTRY = {"mlp": MLPDecoder, "deep_mlp": DeepMLPDecoder}


def build_decoder(name: str, K: int, T: int, **kwargs) -> nn.Module:
    if name not in DECODER_REGISTRY:
        raise KeyError(f"unknown decoder '{name}'. choices: {list(DECODER_REGISTRY)}")
    return DECODER_REGISTRY[name](K=K, T=T, **kwargs)
