"""Saliency extraction via vision-tower attention rollout.

Implements Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers"
rollout. We re-use the SigLIP-style vision encoder that already sits inside
BAGEL — no extra forward pass is needed beyond a single call that returns
`attentions=True`. The CLS-token row of the rollout matrix gives a per-patch
importance scalar that we treat as the saliency map.

This module is independent of BAGEL: it accepts a list of attention tensors
in the shape (B, H, T, T) and returns a saliency tensor of shape (B, T-1)
covering only the non-CLS tokens (patches).
"""

from __future__ import annotations

import torch

__all__ = ["attention_rollout", "saliency_from_attentions"]


def attention_rollout(
    attentions: list[torch.Tensor],
    *,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> torch.Tensor:
    """Recursive rollout per Abnar & Zuidema (2020).

    Args:
        attentions: list of L tensors, each shape (B, H, T, T), per-layer
            attention weights from a transformer.
        head_fusion: how to fuse heads — "mean", "min", or "max".
        discard_ratio: optional bottom-fraction of edges to zero out per
            layer before re-normalising. 0.0 = vanilla rollout.

    Returns:
        Rolled-up attention of shape (B, T, T).
    """
    if not attentions:
        raise ValueError("attentions must be a non-empty list")

    B, _, T, T2 = attentions[0].shape
    if T != T2:
        raise ValueError(
            f"each attention must be square in last two dims, got {attentions[0].shape}"
        )

    device = attentions[0].device
    dtype = attentions[0].dtype

    rollout = torch.eye(T, device=device, dtype=dtype).expand(B, T, T).clone()

    for attn in attentions:
        if attn.shape[0] != B or attn.shape[-1] != T:
            raise ValueError(
                f"inconsistent attention shape: {attn.shape}, expected (B={B}, H, T={T}, T)"
            )
        if head_fusion == "mean":
            fused = attn.mean(dim=1)
        elif head_fusion == "min":
            fused = attn.min(dim=1).values
        elif head_fusion == "max":
            fused = attn.max(dim=1).values
        else:
            raise ValueError(f"unknown head_fusion {head_fusion!r}")

        if discard_ratio > 0.0:
            k = int(T * discard_ratio)
            if k > 0:
                # Per-row threshold: discard the bottom-k entries of each query row.
                threshold = fused.kthvalue(k, dim=-1, keepdim=True).values
                fused = torch.where(fused >= threshold, fused, torch.zeros_like(fused))

        identity = torch.eye(T, device=device, dtype=dtype)
        augmented = fused + identity
        augmented = augmented / augmented.sum(dim=-1, keepdim=True).clamp(min=1e-9)

        rollout = augmented @ rollout

    return rollout


def saliency_from_attentions(
    attentions: list[torch.Tensor],
    *,
    cls_index: int = 0,
    normalize: bool = True,
    head_fusion: str = "mean",
) -> torch.Tensor:
    """Extract a per-token saliency vector from attention rollout.

    Args:
        attentions: list of (B, H, T, T) per-layer attentions.
        cls_index: index of the CLS / register token whose row in the
            rollout matrix is used as the saliency source. Default 0.
        normalize: if True, scale the saliency to [0, 1] per batch.
        head_fusion: forwarded to `attention_rollout`.

    Returns:
        saliency tensor of shape (B, T-1) covering non-CLS tokens.
    """
    if cls_index < 0:
        raise ValueError("cls_index must be >= 0")

    rollout = attention_rollout(attentions, head_fusion=head_fusion)
    _, T, _ = rollout.shape
    if cls_index >= T:
        raise ValueError(f"cls_index {cls_index} out of range for T={T}")

    cls_row = rollout[:, cls_index, :]
    keep = torch.ones(T, dtype=torch.bool, device=rollout.device)
    keep[cls_index] = False
    saliency = cls_row[:, keep]

    if normalize:
        s_max = saliency.amax(dim=-1, keepdim=True)
        s_min = saliency.amin(dim=-1, keepdim=True)
        denom = s_max - s_min
        # If a row is all-equal, fall back to a neutral 0.5 saliency rather
        # than degenerating to zeros (which would silently disable SBSR bias).
        neutral = torch.full_like(saliency, 0.5)
        saliency = torch.where(denom > 1e-9, (saliency - s_min) / denom.clamp(min=1e-9), neutral)

    return saliency
