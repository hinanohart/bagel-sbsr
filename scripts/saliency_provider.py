"""SigLIP2 rollout saliency provider (opt-in; default path is LatentMagnitudeProvider).

BAGEL upstream's SigLIP self-attention uses flash-attn2 and does *not* return
attention weights, so rollout-based saliency requires patching SigLIP to use
an `eager` / SDPA attention implementation that emits `attn_weights`.

This provider supports two operating modes:
  1. `mode="hook"` (default): forward-hook on each SigLIP layer's `self_attn`
     captures attention if and only if the layer's forward returns a 2-tuple
     `(out, attn)`. With stock BAGEL this stays empty and we fall back to
     uniform saliency (0.5), with a one-time warning.
  2. `mode="strict"`: raises at first forward if no attention was captured.
     Use when you have manually patched SigLIP to expose attentions.

Default training (S1/S2/S3) uses `bagel_sbsr.LatentMagnitudeProvider` instead
(no extra forward, no SigLIP patching required). This file is kept for the
v0.2 SigLIP-rollout track.
"""

from __future__ import annotations

import sys
import warnings
import weakref
from pathlib import Path

import torch

from bagel_sbsr.saliency import saliency_from_attentions


def _ensure_vendor_on_path(vendor: str | Path = "vendor/bagel-upstream") -> None:
    vp = str(Path(vendor).resolve())
    if vp not in sys.path:
        sys.path.insert(0, vp)


class SigLIP2RolloutProvider:
    """Forward-hook-based saliency provider (SigLIP2 attention rollout).

    Mapping caveat (v0.1.0): the ViT patch grid does not align 1:1 with the
    BAGEL packed gen-token sequence in the general case. This provider
    currently emits a *placeholder* mapping (first n_target patches) and
    warns at instantiation. Callers who need precise alignment should
    implement a `(latent_patch_row, latent_patch_col) -> vit_patch_index`
    look-up table against the actual VAE / ViT patch geometry of their run.
    """

    def __init__(
        self,
        bagel_model,
        *,
        head_fusion: str = "mean",
        mode: str = "hook",
    ) -> None:
        if mode not in {"hook", "strict"}:
            raise ValueError(f"mode must be 'hook' or 'strict'; got {mode!r}")
        _ensure_vendor_on_path()
        try:
            from modeling.bagel import SiglipVisionModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "SigLIP2RolloutProvider requires vendor/bagel-upstream/ on PYTHONPATH"
            ) from e

        vit = getattr(bagel_model, "vit_model", None)
        if not isinstance(vit, SiglipVisionModel):
            raise TypeError(
                f"bagel_model.vit_model is not SiglipVisionModel; got {type(vit).__name__}"
            )

        warnings.warn(
            "SigLIP2RolloutProvider: stock BAGEL SigLIP uses flash-attn and does "
            "NOT return attention weights. This provider will fall back to "
            "neutral saliency unless SigLIP is patched. Default training uses "
            "bagel_sbsr.LatentMagnitudeProvider instead.",
            RuntimeWarning,
            stacklevel=2,
        )

        self._model_ref = weakref.ref(bagel_model)
        self._head_fusion = head_fusion
        self._mode = mode
        self._captured: list[torch.Tensor] = []
        self._handles: list = []

        for layer in vit.vision_model.encoder.layers:
            h = layer.self_attn.register_forward_hook(self._capture_attn)
            self._handles.append(h)

    def _capture_attn(self, _module, _inputs, output) -> None:
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self._captured.append(output[1].detach())

    def reset(self) -> None:
        """Clear captured attentions (call before every new batch's forward)."""
        self._captured.clear()

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __call__(
        self,
        packed_sequence_gen: torch.Tensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        n_target = packed_sequence_gen.shape[0]
        device = packed_sequence_gen.device

        if not self._captured:
            if self._mode == "strict":
                raise RuntimeError("SigLIP2RolloutProvider strict mode: no attentions captured.")
            return torch.full((n_target,), 0.5, device=device)

        try:
            sal = saliency_from_attentions(
                self._captured, cls_index=0, normalize=True, head_fusion=self._head_fusion
            )
        finally:
            # Discard captures after consumption to avoid memory leak.
            self._captured.clear()

        flat = sal.reshape(-1).to(device)
        if flat.numel() >= n_target:
            return flat[:n_target]
        pad = torch.full((n_target - flat.numel(),), 0.5, device=device)
        return torch.cat([flat, pad])
