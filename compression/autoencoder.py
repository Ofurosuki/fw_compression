"""Encoder -> latent -> decoder autoencoder, plus the training loss."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from compression.decoders import build_decoder
from compression.encoders import build_encoder


class WaveformAutoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    @property
    def T(self) -> int:
        return self.encoder.T

    @property
    def K(self) -> int:
        return self.encoder.K

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z


def build_autoencoder(
    encoder_name: str,
    T: int,
    K: int,
    decoder_name: str = "mlp",
    encoder_kwargs: Optional[dict] = None,
    decoder_kwargs: Optional[dict] = None,
) -> WaveformAutoencoder:
    enc = build_encoder(encoder_name, T=T, K=K, **(encoder_kwargs or {}))
    dec = build_decoder(decoder_name, K=K, T=T, **(decoder_kwargs or {}))
    return WaveformAutoencoder(enc, dec)


def reconstruction_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    energy_weight: float = 0.0,
    peak_weight: float = 0.0,
    peak_mask: Optional[torch.Tensor] = None,
):
    """Reconstruction loss.

    - base: MSE(x_hat, x)
    - optional energy/intensity preservation: |sum(x_hat) - sum(x)| (per-waveform)
    - optional peak-aware term: extra MSE weighted on labelled peak regions.

    ``peak_mask`` (if given) is a [B, T] float mask emphasising peak bins.

    Returns (total_loss, components_dict). Structured so a downstream Ghost-FWL
    loss term can be added later without changing the training loop signature.
    """
    mse = torch.mean((x_hat - x) ** 2)
    total = mse
    comps = {"mse": mse.detach()}

    if energy_weight > 0:
        energy = torch.mean(torch.abs(x_hat.sum(dim=1) - x.sum(dim=1)))
        total = total + energy_weight * energy
        comps["energy"] = energy.detach()

    if peak_weight > 0 and peak_mask is not None:
        denom = peak_mask.sum().clamp_min(1.0)
        peak_term = (((x_hat - x) ** 2) * peak_mask).sum() / denom
        total = total + peak_weight * peak_term
        comps["peak"] = peak_term.detach()

    comps["total"] = total.detach()
    return total, comps
