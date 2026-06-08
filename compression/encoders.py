"""Lightweight sensor-side encoders.

All encoders map a waveform ``x`` of shape ``[B, T]`` to a compressed latent
``z`` of shape ``[B, K]`` with ``K << T``. The design philosophy (see
initial_research_plan.md) is that the encoder must be *cheap* -- something a
sensor could plausibly compute -- so we restrict ourselves to linear / fixed
transforms (binning, random projection, DCT) plus one learnable *linear* map.

The heavy lifting is deferred to the off-sensor DNN decoder.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class BaseEncoder(nn.Module):
    encoder_type: str = "base"

    def __init__(self, T: int, K: int):
        super().__init__()
        self.T = T
        self.K = K

    @property
    def compression_ratio(self) -> float:
        return self.T / self.K


class CoarseBinningEncoder(BaseEncoder):
    """Divide ``T`` into ``K`` (near-)equal bins and average each bin.

    Implemented as a fixed, non-learnable averaging matrix so it works for any
    ``T`` not divisible by ``K``.
    """

    encoder_type = "coarse_binning"

    def __init__(self, T: int, K: int, reduce: str = "mean"):
        super().__init__(T, K)
        assert reduce in ("mean", "sum")
        edges = torch.linspace(0, T, K + 1).round().long()
        W = torch.zeros(K, T)
        for k in range(K):
            lo, hi = int(edges[k]), int(edges[k + 1])
            hi = max(hi, lo + 1)
            if reduce == "mean":
                W[k, lo:hi] = 1.0 / (hi - lo)
            else:
                W[k, lo:hi] = 1.0
        self.register_buffer("W", W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.t()


class RandomProjectionEncoder(BaseEncoder):
    """Fixed random projection ``z = x @ C.T`` with row-normalized ``C``."""

    encoder_type = "random_projection"

    def __init__(self, T: int, K: int, seed: int = 0):
        super().__init__(T, K)
        g = torch.Generator().manual_seed(seed)
        C = torch.randn(K, T, generator=g)
        C = C / (C.norm(dim=1, keepdim=True) + 1e-8)
        self.register_buffer("C", C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.C.t()


class DCTLowFreqEncoder(BaseEncoder):
    """Keep the first ``K`` DCT-II coefficients (low-frequency truncation).

    A fixed orthogonal-ish linear transform; ``z = x @ B.T`` where ``B`` holds
    the first ``K`` DCT basis vectors. This is the classic depth/energy-preserving
    transform and serves as the "depth-oriented compression" baseline.
    """

    encoder_type = "dct_lowfreq"

    def __init__(self, T: int, K: int):
        super().__init__(T, K)
        n = torch.arange(T).float()
        basis = torch.zeros(K, T)
        for k in range(K):
            basis[k] = torch.cos(math.pi / T * (n + 0.5) * k)
            # orthonormal DCT-II scaling
            scale = math.sqrt(1.0 / T) if k == 0 else math.sqrt(2.0 / T)
            basis[k] *= scale
        self.register_buffer("B", basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.B.t()


class LearnableLinearEncoder(BaseEncoder):
    """Learnable linear map ``z = x @ C.T`` with row-normalization in forward.

    Still linear (sensor-friendly) but trained end-to-end with the decoder.
    """

    encoder_type = "learnable_linear"

    def __init__(self, T: int, K: int, seed: int = 0):
        super().__init__(T, K)
        g = torch.Generator().manual_seed(seed)
        C = torch.randn(K, T, generator=g) * (1.0 / math.sqrt(T))
        self.C = nn.Parameter(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = self.C / (self.C.norm(dim=1, keepdim=True) + 1e-8)
        return x @ C.t()


ENCODER_REGISTRY = {
    "coarse_binning": CoarseBinningEncoder,
    "random_projection": RandomProjectionEncoder,
    "dct_lowfreq": DCTLowFreqEncoder,
    "learnable_linear": LearnableLinearEncoder,
}


def build_encoder(name: str, T: int, K: int, **kwargs) -> BaseEncoder:
    if name not in ENCODER_REGISTRY:
        raise KeyError(f"unknown encoder '{name}'. choices: {list(ENCODER_REGISTRY)}")
    return ENCODER_REGISTRY[name](T=T, K=K, **kwargs)
