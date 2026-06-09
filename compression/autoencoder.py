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
    bg_weight: float = 0.0,
    fp_weight: float = 0.0,
    bg_mask: Optional[torch.Tensor] = None,
):
    """Reconstruction loss.

    - base: MSE(x_hat, x)
    - optional energy/intensity preservation: |sum(x_hat) - sum(x)| (per-waveform)
    - optional peak-aware term: extra MSE weighted on labelled peak regions.
    - optional **anti-hallucination** terms (suppress spurious / nonexistent peaks):
        * ``bg_weight`` -- background over-shoot suppression. Penalises
          ``relu(x_hat - x)`` (signal reconstructed *above* the truth) in non-peak
          bins. Asymmetric on purpose: it kills smear / fake bumps but does NOT
          force a genuine non-zero baseline or EMG tail to zero (those are part of
          ``x``, so they incur no penalty).
        * ``fp_weight`` -- a differentiable false-peak surrogate. ``relu(slope_left)
          * relu(slope_right)`` is > 0 only at a local maximum and grows with its
          prominence; restricted to background bins it directly punishes spurious
          peaks without touching true ones.

    ``peak_mask`` / ``bg_mask`` (if given) are [B, T] float masks; ``bg_mask`` marks
    the background (non-peak) bins and is typically ``1 - dilated(peak_mask)``.

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

    if bg_weight > 0 and bg_mask is not None:
        denom = bg_mask.sum().clamp_min(1.0)
        overshoot = torch.relu(x_hat - x)
        bg_term = ((overshoot ** 2) * bg_mask).sum() / denom
        total = total + bg_weight * bg_term
        comps["bg"] = bg_term.detach()

    if fp_weight > 0 and bg_mask is not None:
        # soft local-max detector on interior bins: positive only at peaks
        dl = torch.relu(x_hat[:, 1:-1] - x_hat[:, :-2])
        dr = torch.relu(x_hat[:, 1:-1] - x_hat[:, 2:])
        peakness = dl * dr
        m = bg_mask[:, 1:-1]
        denom = m.sum().clamp_min(1.0)
        fp_term = (peakness * m).sum() / denom
        total = total + fp_weight * fp_term
        comps["fp"] = fp_term.detach()

    comps["total"] = total.detach()
    return total, comps
