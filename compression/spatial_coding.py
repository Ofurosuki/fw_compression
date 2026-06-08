"""Spatio-temporal compressive coding (ICCV2023-style), as the spatial counterpart
to the per-pixel ``encoders.py`` / ``decoders.py``.

A local ``P = Mr*Mc`` block of neighbouring pixel waveforms ``x`` (shape ``[B, P, T]``)
is compressed jointly into ``z`` of shape ``[B, K]`` by ``K`` **separable** coding
tensors ``C_k = c^t_k ⊗ c^s_k`` (temporal ``c^t_k`` of length ``T`` ⊗ spatial
``c^s_k`` of length ``P``), following the paper's separable design that keeps the
in-pixel parameter count small (``K*(T+P)`` instead of ``K*T*P``):

    z[b,k] = Σ_{p,t} c^s_k[p] c^t_k[t] x[b,p,t]
           = Σ_p c^s_k[p] ( x[b,p,:] · c^t_k )

The (off-sensor, heavy) decoder maps ``z -> x_hat`` of the full block ``[B, P, T]``.
The implied per-pixel compression ratio is ``(P*T)/K``; comparing against the
per-pixel encoder at a *matched* ratio isolates the benefit of spatial context.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SpatialSeparableEncoder(nn.Module):
    """Learned separable spatio-temporal coding tensors ``C_k = c^t_k ⊗ c^s_k``.

    Each ``C_k`` is row-normalized (unit Frobenius norm of the separable tensor =
    product of the two factor norms), mirroring the row-normalization of the per-pixel
    ``LearnableLinearEncoder`` so the latent scale is comparable.
    """

    encoder_type = "spatial_separable"

    def __init__(self, T: int, K: int, P: int = 16, seed: int = 0):
        super().__init__()
        self.T = T
        self.K = K
        self.P = P
        g = torch.Generator().manual_seed(seed)
        self.Ct = nn.Parameter(torch.randn(K, T, generator=g) * (1.0 / math.sqrt(T)))  # temporal
        self.Cs = nn.Parameter(torch.randn(K, P, generator=g) * (1.0 / math.sqrt(P)))  # spatial

    @property
    def compression_ratio(self) -> float:
        return (self.P * self.T) / self.K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, P, T]
        ct = self.Ct / (self.Ct.norm(dim=1, keepdim=True) + 1e-8)   # [K, T]
        cs = self.Cs / (self.Cs.norm(dim=1, keepdim=True) + 1e-8)   # [K, P]
        # temporal projection per pixel: [B, P, K]
        tp = torch.einsum("bpt,kt->bpk", x, ct)
        # spatial mix: [B, K]
        z = torch.einsum("bpk,kp->bk", tp, cs)
        return z


class SpatialMLPDecoder(nn.Module):
    """Off-sensor DNN decoder ``z[B,K] -> x_hat[B, P, T]`` (non-negative)."""

    def __init__(self, K: int, T: int, P: int = 16, hidden=(512, 1024), nonneg: bool = True):
        super().__init__()
        self.T = T
        self.P = P
        dims = [K, *hidden, P * T]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self.nonneg = nonneg

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z).view(z.shape[0], self.P, self.T)
        if self.nonneg:
            x = torch.relu(x)
        return x


class SpatialAutoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @property
    def T(self):
        return self.encoder.T

    @property
    def K(self):
        return self.encoder.K

    @property
    def P(self):
        return self.encoder.P

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def build_spatial_autoencoder(T: int, K: int, P: int = 16, seed: int = 0) -> SpatialAutoencoder:
    enc = SpatialSeparableEncoder(T=T, K=K, P=P, seed=seed)
    dec = SpatialMLPDecoder(K=K, T=T, P=P)
    return SpatialAutoencoder(enc, dec)
