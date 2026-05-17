"""Concrete saliency_provider for `bagel_sbsr.patch_bagel`.

Captures the BAGEL SigLIP2 vision tower's per-layer attentions via a forward
hook, then runs `saliency_from_attentions` to produce a per-patch saliency
vector. Designed to be called from the SBSR attention hook on every
gen-expert forward.

This sits outside src/bagel_sbsr/ because it imports BAGEL upstream
(modeling.bagel.SiglipVisionModel) and so cannot live in the always-importable
core package.
"""

from __future__ import annotations

import sys
import weakref
from pathlib import Path

import torch

from bagel_sbsr.saliency import saliency_from_attentions


def _ensure_vendor_on_path(vendor: str | Path = "vendor/bagel-upstream") -> None:
    vp = str(Path(vendor).resolve())
    if vp not in sys.path:
        sys.path.insert(0, vp)


class SigLIP2RolloutProvider:
    """Forward-hook-based saliency provider.

    Usage:
        provider = SigLIP2RolloutProvider(bagel_model)
        patch_bagel(bagel_model, saliency_provider=provider, ...)

    The provider attaches a forward hook on the vision tower that captures
    per-layer attention weights at every forward pass; subsequent SBSR calls
    consume the most recent capture.
    """

    def __init__(self, bagel_model, *, head_fusion: str = "mean") -> None:
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

        self._model_ref = weakref.ref(bagel_model)
        self._head_fusion = head_fusion
        self._captured: list[torch.Tensor] = []

        for layer in vit.vision_model.encoder.layers:
            layer.self_attn.register_forward_hook(self._capture_attn)

    def _capture_attn(self, module, inputs, output) -> None:
        if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
            self._captured.append(output[1].detach())

    def reset(self) -> None:
        self._captured.clear()

    def __call__(
        self,
        packed_sequence_gen: torch.Tensor,
        packed_gen_token_indexes: torch.LongTensor,
    ) -> torch.Tensor:
        if not self._captured:
            return torch.zeros(packed_sequence_gen.shape[0], device=packed_sequence_gen.device)

        sal = saliency_from_attentions(
            self._captured,
            cls_index=0,
            normalize=True,
            head_fusion=self._head_fusion,
        )

        flat = sal.reshape(-1)
        n_target = packed_sequence_gen.shape[0]
        if flat.numel() >= n_target:
            return flat[:n_target].to(packed_sequence_gen.device)
        pad = torch.zeros(n_target - flat.numel(), device=packed_sequence_gen.device)
        return torch.cat([flat.to(packed_sequence_gen.device), pad])
