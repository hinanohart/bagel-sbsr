"""Install SBSR onto BAGEL's `PackedAttentionMoT` layers.

The hook wraps `PackedAttentionMoT.forward_train` so that on the gen-expert
path (identified by `packed_gen_token_indexes`), an additive SBSR bias is
mixed into the per-sample `scaled_dot_product_attention` attention mask
before softmax. Saliency is supplied by a user-provided callable so this
module remains decoupled from any specific saliency source.

This file imports BAGEL upstream lazily — installing the patch requires
`vendor/bagel-upstream/` to be importable. Without it, `patch_bagel` raises
ImportError at call time but the rest of `bagel_sbsr` remains usable.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
import torch.nn as nn

from .sbsr import SBSR

__all__ = ["BagelNotInstalledError", "patch_bagel"]

SaliencyProvider = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
"""Callable signature: (packed_sequence_gen, packed_gen_token_indexes) -> saliency tensor (T_gen,)."""


class BagelNotInstalledError(ImportError):
    """Raised when BAGEL upstream is not importable from the current PYTHONPATH."""


def _require_bagel() -> type[nn.Module]:
    try:
        from modeling.bagel.qwen2_navit import PackedAttentionMoT  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - exercised only when vendor missing
        raise BagelNotInstalledError(
            "BAGEL upstream is not importable. Run scripts/install_bagel_src.sh and "
            "add vendor/bagel-upstream/ to PYTHONPATH before calling patch_bagel()."
        ) from e
    return PackedAttentionMoT


def patch_bagel(
    model: nn.Module,
    *,
    lambda_init: float = 0.5,
    mu_init: float = 0.25,
    top_k: int | None = 64,
    saliency_provider: SaliencyProvider | None = None,
    learnable: bool = True,
) -> list[SBSR]:
    """Install one shared `SBSR` module across every PackedAttentionMoT layer.

    A single `SBSR` instance is attached to `model.sbsr` (creating it if absent)
    and every PackedAttentionMoT layer's `forward_train` is wrapped so its
    gen-token path receives the SBSR additive bias.

    Args:
        model: the BAGEL model (must have nn.Module attributes containing
            PackedAttentionMoT layers).
        lambda_init: initial value of the SBSR bias coefficient.
        mu_init: initial value of the saliency-difference coefficient.
        top_k: optional sparsification on the gen-path logits.
        saliency_provider: callable returning saliency (T_gen,) given the
            packed gen sequence and gen-token indices. If None, the patch
            installs no bias (zero override, behaves like vanilla BAGEL) —
            useful for dry-run.
        learnable: if False, (lambda, mu) are frozen buffers.

    Returns:
        the list of patched modules (so the caller can introspect or unpatch).
    """
    PackedAttentionMoT = _require_bagel()

    sbsr = getattr(model, "sbsr", None)
    if sbsr is None:
        sbsr = SBSR(
            lambda_init=lambda_init,
            mu_init=mu_init,
            top_k=top_k,
            learnable=learnable,
        )
        model.add_module("sbsr", sbsr)

    patched: list[SBSR] = []
    for _, module in model.named_modules():
        if isinstance(module, PackedAttentionMoT) and not getattr(module, "_sbsr_patched", False):
            _wrap_forward_train(module, sbsr, saliency_provider)
            module._sbsr_patched = True
            patched.append(sbsr)
    return patched


def _wrap_forward_train(
    layer: nn.Module,
    sbsr: SBSR,
    saliency_provider: SaliencyProvider | None,
) -> None:
    original = layer.forward_train

    @functools.wraps(original)
    def wrapped(
        packed_sequence: torch.Tensor,
        sample_lens: list[int],
        attention_mask,
        packed_position_embeddings,
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
    ):
        if saliency_provider is None or packed_gen_token_indexes.numel() == 0:
            return original(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        if not isinstance(attention_mask, list):
            return original(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        sal_full = sbsr.bias.__self__  # for type-hint friendliness; we recompute below
        sal_gen = saliency_provider(
            packed_sequence[packed_gen_token_indexes], packed_gen_token_indexes
        )

        full_saliency = packed_sequence.new_zeros(packed_sequence.shape[0])
        full_saliency[packed_gen_token_indexes] = sal_gen.to(full_saliency.dtype)

        biased_masks: list = []
        offset = 0
        for n, m in zip(sample_lens, attention_mask):
            sample_sal = full_saliency[offset : offset + n]
            offset += n
            bias = sbsr.bias(sample_sal).to(m.dtype)
            biased_masks.append(m + bias)

        return original(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=biased_masks,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )

    layer.forward_train = wrapped  # type: ignore[method-assign]


def unpatch_bagel(model: nn.Module) -> int:
    """Reverse `patch_bagel`. Returns the count of layers restored."""
    try:
        PackedAttentionMoT = _require_bagel()
    except BagelNotInstalledError:
        return 0

    n = 0
    for _, module in model.named_modules():
        if isinstance(module, PackedAttentionMoT) and getattr(module, "_sbsr_patched", False):
            del module.forward_train  # type: ignore[attr-defined]
            module._sbsr_patched = False
            n += 1
    return n
