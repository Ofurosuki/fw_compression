"""SparseEventToPM: a token-native sparse spatial-range network for FW-LiDAR events.

Motivation (see ``eventnet_v4.md``): the ToPM-retrain experiment showed top-K
``(t,a,w)`` event tokens carry ~all the downstream-relevant information, yet
EventNet V2 only reaches ~0.555 vs ToPM-retrain's ~0.582. The hypothesis is that
EventNet **flattens the K events of each ray into channels** (``reshape(... K*emb)``)
*before* the spatial U-Net, collapsing the event/range axis too early.

This model keeps every event as a **token with coordinate ``(h, w, k)``** all the
way through the spatial mixing: attention runs jointly over the
``window x window x K`` tokens inside local spatial windows, so an event at pixel
``(h,w)`` depth ``t_i`` can attend to an event at a *neighbouring* pixel depth
``t_j`` without ever being merged into channels. Shifted windows (implemented via
a padded partition offset, so each window is a contiguous region needing only a
padding mask — no Swin cyclic-shift region mask) connect window boundaries and
grow the receptive field across depth.

Input  ``events: [B, H, W, K, F]`` (+ ``valid: [B, H, W, K]``)
Output ``logits: [B, H, W, K, num_classes]``

Drop-in with the existing eventnet harness: ``build_model(arch='setopm', ...)``,
trained by ``eventnet/train.py`` and scored by ``eventnet/evaluate.py`` (paper
peak-level F1), so the only thing that differs from the V2 comparison is the
architecture.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# --------------------------------------------------------------------------- #
# Windowed (shifted) spatial-event attention
# --------------------------------------------------------------------------- #
def _partition(x, valid, ws, shift, t=None):
    """Partition ``x [B,H,W,K,C]`` into non-overlapping ``ws x ws`` spatial windows,
    keeping the K events of every spatial cell as separate tokens.

    A ``shift`` (0 or ws//2) is realised by padding ``shift`` rows/cols at the
    top-left (and the remainder at the bottom-right); padded cells are marked
    invalid. Because the partition never wraps around, each window is a single
    contiguous image region, so only a padding (validity) mask is needed — no
    Swin cyclic-shift region mask.

    Returns ``win [N, ws*ws*K, C]``, ``vmask [N, ws*ws*K]`` (bool, True=valid),
    ``twin [N, ws*ws*K]`` (per-token range coord, or None), and ``info``.
    """
    B, H, W, K, C = x.shape
    pad_t = pad_l = shift
    Hp = math.ceil((H + shift) / ws) * ws
    Wp = math.ceil((W + shift) / ws) * ws
    pad_b, pad_r = Hp - H - pad_t, Wp - W - pad_l
    # F.pad on [B,H,W,K,C]: dims ordered C,K,W,H from the last -> pad H and W.
    xpad = F.pad(x, (0, 0, 0, 0, pad_l, pad_r, pad_t, pad_b))
    vpad = F.pad(valid, (0, 0, pad_l, pad_r, pad_t, pad_b), value=False)
    nh, nw = Hp // ws, Wp // ws
    xw = xpad.view(B, nh, ws, nw, ws, K, C).permute(0, 1, 3, 2, 4, 5, 6).contiguous()
    vw = vpad.view(B, nh, ws, nw, ws, K).permute(0, 1, 3, 2, 4, 5).contiguous()
    win = xw.view(B * nh * nw, ws * ws * K, C)
    vmask = vw.view(B * nh * nw, ws * ws * K)
    twin = None
    if t is not None:
        tpad = F.pad(t, (0, 0, pad_l, pad_r, pad_t, pad_b), value=0.0)
        tw = tpad.view(B, nh, ws, nw, ws, K).permute(0, 1, 3, 2, 4, 5).contiguous()
        twin = tw.view(B * nh * nw, ws * ws * K)
    return win, vmask, twin, (B, H, W, K, C, nh, nw, ws, pad_t, pad_l)


def _reverse(win, info):
    """Inverse of ``_partition`` -> ``x [B,H,W,K,C]``."""
    B, H, W, K, C, nh, nw, ws, pad_t, pad_l = info
    xw = win.view(B, nh, nw, ws, ws, K, C).permute(0, 1, 3, 2, 4, 5, 6).contiguous()
    xpad = xw.view(B, nh * ws, nw * ws, K, C)
    return xpad[:, pad_t:pad_t + H, pad_l:pad_l + W]


def _rope_range(q, k, twin, dim_half, base=10000.0, scale=300.0):
    """Rotary position embedding on the RANGE coordinate. Rotating q,k by an angle
    proportional to each token's range ``t`` makes the dot product q_i·k_j depend
    on the relative gap ``t_i - t_j`` — translation-equivariant in range, the
    token-native analog of ToPM's Conv3d-over-depth, and (unlike an explicit T×T
    bias) it keeps the fast SDPA kernel since attention still gets only the cheap
    padding mask. ``q,k``: [N,heads,T,hd]; ``twin``: [N,T] in [0,1]."""
    N, Hd, T, hd = q.shape
    d = torch.arange(dim_half, device=q.device, dtype=torch.float32)
    freqs = base ** (-d / dim_half)                          # [hd/2]
    ang = (twin.float() * scale)[:, None, :, None] * freqs   # [N,1,T,hd/2]
    cos, sin = ang.cos().to(q.dtype), ang.sin().to(q.dtype)

    def rot(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]                  # [N,heads,T,hd/2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return rot(q), rot(k)


class WindowEventAttention(nn.Module):
    """Multi-head self-attention over the ``ws*ws*K`` tokens of one window, with a
    learned within-window positional embedding (spatial cell + event rank), an
    optional relative-range RoPE, and a validity mask so padded / non-existent
    events are never attended to."""

    def __init__(self, dim, window, K, heads, rel_range=False):
        super().__init__()
        self.dim, self.window, self.K, self.heads = dim, window, K, heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # within-window position: per spatial cell (ws*ws) + per event rank (K)
        self.spatial_pos = nn.Parameter(torch.zeros(window * window, dim))
        self.rank_pos = nn.Parameter(torch.zeros(K, dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.rank_pos, std=0.02)
        self.rel_range = rel_range

    def forward(self, win, vmask, twin=None):
        N, T, C = win.shape                                  # T = ws*ws*K
        pos = (self.spatial_pos[:, None, :] + self.rank_pos[None, :, :]).reshape(T, C)
        x = win + pos[None]
        qkv = self.qkv(x).reshape(N, T, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)        # [N,heads,T,hd]
        if self.rel_range and twin is not None:
            q, k = _rope_range(q, k, twin, (C // self.heads) // 2)
        # additive key-padding mask, broadcast over heads & query positions.
        # fully-invalid windows (background): unmask so softmax has no all -inf row
        # (their tokens are invalid and excluded from the loss / metric anyway).
        keep = vmask | (~vmask.any(dim=1, keepdim=True))
        bias = torch.zeros(N, 1, 1, T, dtype=q.dtype, device=q.device)
        bias.masked_fill_(~keep[:, None, None, :], -1e9)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(N, T, C)
        return self.proj(out)


class SpatialEventBlock(nn.Module):
    """Pre-norm windowed spatial-event attention + FFN (Transformer block)."""

    def __init__(self, dim, window, K, heads, ffn_mult, shift, rel_range=False):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowEventAttention(dim, window, K, heads, rel_range=rel_range)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult), nn.GELU(),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(self, x, valid, t=None):
        win, vmask, twin, info = _partition(self.norm1(x), valid, self.window,
                                            self.shift, t=t)
        x = x + _reverse(self.attn(win, vmask, twin), info)
        x = x + self.ffn(self.norm2(x))
        return x


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Hierarchy (v2): spatial down/up-sampling that KEEPS the K events as tokens, so
# glass (a relational/planar class needing large spatial receptive field) gets
# multi-scale context while object/ghost keep their per-event token fidelity.
# --------------------------------------------------------------------------- #
class PatchMergeK(nn.Module):
    """Swin-style 2x2 spatial patch merge applied per event-slot: ``[B,H,W,K,C]``
    -> ``[B,H/2,W/2,K,C]`` (H,W even). The 4 spatial neighbours of each k-slot are
    concatenated into channels and linearly mixed."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.lin = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        B, H, W, K, C = x.shape
        x = x.view(B, H // 2, 2, W // 2, 2, K, C).permute(0, 1, 3, 5, 2, 4, 6)
        x = x.reshape(B, H // 2, W // 2, K, 4 * C)
        return self.lin(self.norm(x))


class PatchExpandK(nn.Module):
    """Nearest 2x upsample per event-slot + linear: ``[B,H,W,K,C]`` -> ``[B,2H,2W,K,C]``."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x = self.lin(self.norm(x))
        return x.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)


class GlobalSpatialContext(nn.Module):
    """Frame-wide spatial context for ghost: pool the K events of each cell into a
    ray summary, run a GLOBAL self-attention over all cells (every ray attends to
    every other), then broadcast that context back onto each event token. Targets
    the finding that dropped ghosts form spatially-coherent regions/bands and need
    3D-context (not per-ray) reasoning — the analog of EventNet V2's spatial_attn,
    the single lever that most moved ghost there. Applied at the (small) bottleneck
    resolution so the global O(N^2) attention is cheap."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, valid):
        B, H, W, K, C = x.shape
        m = valid.float().unsqueeze(-1)
        cell = (x * m).sum(3) / m.sum(3).clamp_min(1.0)        # [B,H,W,C] mean over valid events
        t = self.norm(cell).reshape(B, H * W, C)
        qkv = self.qkv(t).reshape(B, H * W, 3, self.heads, C // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)          # [B,heads,N,hd]
        ctx = F.scaled_dot_product_attention(q, k, v)
        ctx = ctx.transpose(1, 2).reshape(B, H, W, 1, C)
        return x + self.proj(ctx)                               # broadcast to every event token


def _down_valid(v):
    """Pool a validity mask 2x2 spatially (per k): coarse cell valid if any fine."""
    B, H, W, K = v.shape
    return v.view(B, H // 2, 2, W // 2, 2, K).permute(0, 1, 3, 5, 2, 4).reshape(
        B, H // 2, W // 2, K, 4).any(-1)


def _up_valid(v):
    return v.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)


def _down_t(t, v):
    """Pool the per-token range coord 2x2 spatially (per k): valid-weighted mean,
    so a coarse cell carries the mean range of its fine valid events."""
    B, H, W, K = t.shape
    tv = t.view(B, H // 2, 2, W // 2, 2, K).permute(0, 1, 3, 5, 2, 4).reshape(B, H // 2, W // 2, K, 4)
    vv = v.view(B, H // 2, 2, W // 2, 2, K).permute(0, 1, 3, 5, 2, 4).reshape(B, H // 2, W // 2, K, 4).float()
    return (tv * vv).sum(-1) / vv.sum(-1).clamp_min(1.0)


def _up_t(t):
    return t.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)


class SparseEventToPMHier(nn.Module):
    """U-shaped windowed spatial-event attention over (h,w,k) event tokens.

    Same token-native premise as ``SparseEventToPM`` (the K events of each ray
    stay tokens, never flattened into channels), but with a symmetric encoder/
    decoder that downsamples the spatial plane (``PatchMergeK``) so deeper blocks
    see a large receptive field — the fix for v1's glass collapse (v1 had no
    spatial hierarchy, RF ~ depth/2 x window). Skip connections preserve the
    fine-resolution per-event detail object/ghost rely on.
    """

    def __init__(self, K, in_dim, embed_dim=256, levels=3, blocks_per_stage=2,
                 num_heads=4, window_size=8, ffn_mult=2, num_classes=4,
                 use_checkpoint=True, global_bottleneck=False, rel_range_bias=False):
        super().__init__()
        self.K = K
        self.embed_dim = embed_dim
        self.levels = levels
        self.use_checkpoint = use_checkpoint
        self.rel_range_bias = rel_range_bias
        self.gctx = GlobalSpatialContext(embed_dim, num_heads) if global_bottleneck else None
        self.event_mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
        )
        self.rank_embedding = nn.Embedding(K, embed_dim)
        self.time_proj = nn.Linear(embed_dim, embed_dim)

        def stage(n):
            return nn.ModuleList([
                SpatialEventBlock(embed_dim, window_size, K, num_heads, ffn_mult,
                                  shift=(window_size // 2) if (i % 2 == 1) else 0,
                                  rel_range=rel_range_bias)
                for i in range(n)])

        self.enc_stages = nn.ModuleList([stage(blocks_per_stage) for _ in range(levels - 1)])
        self.downs = nn.ModuleList([PatchMergeK(embed_dim) for _ in range(levels - 1)])
        self.bottleneck = stage(blocks_per_stage)
        self.ups = nn.ModuleList([PatchExpandK(embed_dim) for _ in range(levels - 1)])
        self.dec_stages = nn.ModuleList([stage(blocks_per_stage) for _ in range(levels - 1)])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def _run(self, blocks, x, valid, t=None):
        for blk in blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, valid, t, use_reentrant=False)
            else:
                x = blk(x, valid, t)
        return x

    def forward(self, events, valid):
        # Autocast INSIDE forward (not just in the caller) so nn.DataParallel
        # replica threads also run bf16 — the efficient SDPA kernel is fp32-disabled
        # on this build. Checkpoint preserves the autocast state on recompute.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=events.is_cuda):
            B, H, W, K, _ = events.shape
            assert K == self.K
            x = self.event_mlp(events)
            rank = self.rank_embedding(torch.arange(K, device=events.device))
            x = x + rank.view(1, 1, 1, K, self.embed_dim)
            t0 = events[..., 0]                               # normalised range coord per token
            x = x + self.time_proj(_sinusoidal(t0, self.embed_dim))

            # pad spatial dims up to a multiple of 2**(levels-1) so every merge is exact
            m = 2 ** (self.levels - 1)
            ph, pw = (-H) % m, (-W) % m
            if ph or pw:
                x = F.pad(x, (0, 0, 0, 0, 0, pw, 0, ph))
                valid = F.pad(valid, (0, 0, 0, pw, 0, ph), value=False)
                t0 = F.pad(t0, (0, 0, 0, pw, 0, ph), value=0.0)
            t = t0 if self.rel_range_bias else None

            skips = []
            v = valid
            for enc, down in zip(self.enc_stages, self.downs):
                x = self._run(enc, x, v, t)
                skips.append((x, v, t))                        # save fine-res x/v/t
                x = down(x)
                if t is not None:
                    t = _down_t(t, v)
                v = _down_valid(v)
            x = self._run(self.bottleneck, x, v, t)
            if self.gctx is not None:
                x = self.gctx(x, v)
            for up, dec, (xs, vs, ts) in zip(self.ups, self.dec_stages, reversed(skips)):
                x = up(x) + xs
                v, t = vs, ts
                x = self._run(dec, x, v, t)

            x = self.norm(x)
            logits = self.classifier(x)
            if ph or pw:
                logits = logits[:, :H, :W]
        return logits.float()


def _sinusoidal(t_norm, dim):
    """Sinusoidal embedding of a scalar in [0,1] -> [..., dim]."""
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=t_norm.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, half - 1)))
    ang = t_norm[..., None] * freqs * (2 * math.pi)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if emb.shape[-1] < dim:                                  # odd dim pad
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class SparseEventToPM(nn.Module):
    """Token-native windowed spatial-event attention over (h,w,k) event tokens.

    The K events of each ray stay as tokens through every block (never flattened
    into channels), and attention mixes them jointly with the events of nearby
    rays inside local (shifted) windows — the hypothesised fix for EventNet's
    early event/channel collapse.
    """

    def __init__(self, K, in_dim, embed_dim=192, depth=12, num_heads=4,
                 window_size=8, ffn_mult=2, num_classes=4, use_checkpoint=True):
        super().__init__()
        self.K = K
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.event_mlp = nn.Sequential(
            nn.Linear(in_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
        )
        self.rank_embedding = nn.Embedding(K, embed_dim)
        self.time_proj = nn.Linear(embed_dim, embed_dim)     # project sinusoidal(t)
        self.blocks = nn.ModuleList([
            SpatialEventBlock(embed_dim, window_size, K, num_heads, ffn_mult,
                              shift=(window_size // 2) if (i % 2 == 1) else 0)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, events, valid):
        """events [B,H,W,K,F], valid [B,H,W,K] -> logits [B,H,W,K,num_classes]."""
        B, H, W, K, _ = events.shape
        assert K == self.K
        x = self.event_mlp(events)                            # [B,H,W,K,C]
        rank = self.rank_embedding(torch.arange(K, device=events.device))
        x = x + rank.view(1, 1, 1, K, self.embed_dim)
        # column 0 of the assembled feature tensor is always normalised t (t/T)
        x = x + self.time_proj(_sinusoidal(events[..., 0], self.embed_dim))
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, valid, use_reentrant=False)
            else:
                x = blk(x, valid)
        x = self.norm(x)
        return self.classifier(x)
