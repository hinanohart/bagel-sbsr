"""Saliency from FLUX VAE latent magnitude.

Rationale (honest scope, v0.1.0):
The original SBSR design (project_oss-architecture-final-2026-05-17) cited
SigLIP2 CLS attention rollout as the saliency source. BAGEL upstream ships
SigLIP with flash-attn-only attention layers that do NOT return attention
weights (`/tmp/bagel-probe/modeling/bagel/siglip_navit.py`), so rollout is
not directly recoverable without patching SigLIP itself. To keep v0.1.0
shippable, SBSR defaults to a saliency source that needs no extra forward
pass: the per-patch L2 magnitude of the FLUX VAE latent.

The latent is already produced by BAGEL's encoder (`load_ae` + transform) on
every training step, so this saliency is computed on tensors already in
memory. It is a *weaker* proxy than rollout — strong-magnitude regions
empirically correlate with image content rather than uniform background —
but it preserves the SBSR contract:
    saliency: [0, 1] per gen token
    higher = the patch carries more visual "energy"

A SigLIP-based provider is also offered (SigLIP2RolloutProvider) for users
who patch SigLIP to expose attentions; that path is opt-in and unused by
the default training recipe.
"""

from __future__ import annotations

import torch

__all__ = ["LatentMagnitudeProvider", "latent_magnitude_saliency"]


def latent_magnitude_saliency(
    latent: torch.Tensor,
    *,
    eps: float = 1e-9,
) -> torch.Tensor:
    """L2-magnitude saliency over the channel dim of a VAE latent.

    Args:
        latent: (B, C, H, W) FLUX VAE latent (C=16 for FLUX).

    Returns:
        Per-patch saliency of shape (B, H * W) in [0, 1] (min-max normalized
        per batch row; constant rows fall back to 0.5).
    """
    if latent.dim() != 4:
        raise ValueError(f"latent must be (B, C, H, W); got shape {tuple(latent.shape)}")
    B, _, H, W = latent.shape
    mag = latent.float().pow(2).sum(dim=1).sqrt().view(B, H * W)
    s_max = mag.amax(dim=-1, keepdim=True)
    s_min = mag.amin(dim=-1, keepdim=True)
    denom = s_max - s_min
    neutral = torch.full_like(mag, 0.5)
    return torch.where(denom > eps, (mag - s_min) / denom.clamp(min=eps), neutral)


class LatentMagnitudeProvider:
    """Saliency provider that maps VAE-latent magnitude onto packed gen tokens.

    Usage:
        provider = LatentMagnitudeProvider(latent_grid=(H, W))
        provider.set_latent(latent_tensor)   # call before every forward
        patch_bagel(model, saliency_provider=provider, ...)

    The caller is responsible for invoking `set_latent(...)` with the VAE
    latent of the *current* batch before each forward, and `reset()` if the
    batch is dropped (e.g. NaN detected).
    """

    def __init__(self, latent_grid: tuple[int, int] | None = None) -> None:
        self.latent_grid = latent_grid
        self._latent: torch.Tensor | None = None
        self._saliency: torch.Tensor | None = None

    def set_latent(self, latent: torch.Tensor) -> None:
        self._latent = latent
        self._saliency = latent_magnitude_saliency(latent)

    def reset(self) -> None:
        self._latent = None
        self._saliency = None

    def __call__(
        self,
        packed_sequence_gen: torch.Tensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        n_target = packed_sequence_gen.shape[0]
        device = packed_sequence_gen.device

        if self._saliency is None:
            # No latent set yet (e.g. first dry-run step) — neutral 0.5.
            return torch.full((n_target,), 0.5, device=device)

        flat = self._saliency.reshape(-1).to(device)
        if flat.numel() >= n_target:
            return flat[:n_target]
        pad = torch.full((n_target - flat.numel(),), 0.5, device=device)
        return torch.cat([flat, pad])
