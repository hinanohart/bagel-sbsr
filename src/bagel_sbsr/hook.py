"""Install SBSR onto BAGEL's `PackedAttentionMoT` layers.

The hook wraps `PackedAttentionMoT.forward_train` so that on the gen-expert
path, an additive SBSR bias + top-k mask are mixed into the per-sample
attention mask before softmax. Saliency is supplied by a user-provided
callable so this module remains decoupled from any specific saliency source.

Scope (honest):
- Training path only. Inference (`forward_inference`, flash_attn_varlen) is
  *not* patched in v0.1.0 — top-k speedups at inference are a v0.2 target.
- SDPA branch only (List-form `attention_mask`). The flex_attention branch
  emits a one-time warning if reached; train_s1.py asserts the List form.

This file imports BAGEL upstream lazily — installing the patch requires
`vendor/bagel-upstream/` to be importable. Without it, `patch_bagel` raises
ImportError at call time but the rest of `bagel_sbsr` remains usable.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable

import torch
import torch.nn as nn

from .sbsr import SBSR

__all__ = ["BagelNotInstalledError", "SaliencyProvider", "patch_bagel", "unpatch_bagel"]

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


def _resolve_root(model: nn.Module) -> nn.Module:
    """Walk through PEFT/Accelerate wrappers to the underlying nn.Module."""
    seen = set()
    node = model
    while id(node) not in seen:
        seen.add(id(node))
        for attr in ("base_model", "model", "module"):
            child = getattr(node, attr, None)
            if isinstance(child, nn.Module) and child is not node:
                node = child
                break
        else:
            break
    return node


def patch_bagel(
    model: nn.Module,
    *,
    lambda_init: float = 0.5,
    mu_init: float = 0.25,
    top_k: int | None = 64,
    saliency_provider: SaliencyProvider | None = None,
    learnable: bool = True,
    require_layers: int = 1,
) -> list[nn.Module]:
    """Install one shared `SBSR` module across every PackedAttentionMoT layer.

    A single `SBSR` instance is attached to the resolved root model as `.sbsr`
    (creating it only if absent and not already an SBSR instance) and every
    `PackedAttentionMoT` layer's `forward_train` is wrapped so its gen-token
    path receives the SBSR additive bias + top-k mask.

    Args:
        model: the BAGEL model (may be wrapped by PEFT/Accelerate).
        lambda_init / mu_init / top_k / learnable: forwarded to SBSR.
        saliency_provider: callable returning saliency (T_gen,) given the
            packed gen sequence and gen-token indices. If None, the wrapper
            falls through to the original forward_train.
        require_layers: minimum number of layers that must be patched, else
            raise ValueError. Catches the silent-zero case where the wrong
            model is passed in.

    Returns:
        the list of patched PackedAttentionMoT layers.
    """
    PackedAttentionMoT = _require_bagel()
    root = _resolve_root(model)

    existing = getattr(root, "sbsr", None)
    if isinstance(existing, SBSR):
        sbsr = existing
    elif existing is None:
        sbsr = SBSR(lambda_init=lambda_init, mu_init=mu_init, top_k=top_k, learnable=learnable)
        root.add_module("sbsr", sbsr)
    else:
        raise RuntimeError(
            f"model.sbsr already exists and is not an SBSR instance "
            f"(got {type(existing).__name__}); refusing to overwrite."
        )

    patched: list[nn.Module] = []
    for _, module in root.named_modules():
        if isinstance(module, PackedAttentionMoT) and not getattr(module, "_sbsr_patched", False):
            _wrap_forward_train(module, sbsr, saliency_provider)
            module._sbsr_patched = True  # type: ignore[attr-defined]
            patched.append(module)

    if len(patched) < require_layers:
        raise ValueError(
            f"patch_bagel installed SBSR on {len(patched)} layer(s); "
            f"require_layers={require_layers}. Check that the model contains "
            f"PackedAttentionMoT layers and is not already fully wrapped."
        )
    return patched


def _wrap_forward_train(
    layer: nn.Module,
    sbsr: SBSR,
    saliency_provider: SaliencyProvider | None,
) -> None:
    original = layer.forward_train
    # Save the *unbound-equivalent* for later restoration. We store the bound
    # method itself, so unpatch can reinstall it verbatim. functools.wraps
    # would copy __wrapped__ across nested wraps, so we use a private slot.
    layer._sbsr_original_forward_train = original  # type: ignore[attr-defined]

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
            # The flex_attention branch consumes a BlockMask, not an additive
            # mask. Wiring SBSR there requires a flex_attention score_mod;
            # deferred to v0.2. Emit a single warning per layer per process.
            if not getattr(layer, "_sbsr_flex_warned", False):
                warnings.warn(
                    "SBSR: flex_attention branch reached but SBSR is bias-only "
                    "on SDPA. Force-disable flex_attention or assert List-form "
                    "attention_mask in your training entrypoint.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                layer._sbsr_flex_warned = True  # type: ignore[attr-defined]
            return original(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )

        sal_gen = saliency_provider(
            packed_sequence[packed_gen_token_indexes], packed_gen_token_indexes
        )

        full_saliency = packed_sequence.new_zeros(packed_sequence.shape[0])
        full_saliency[packed_gen_token_indexes] = sal_gen.to(full_saliency.dtype)

        # Build a gen-only marker so the additive bias affects only rows/cols
        # that are gen tokens; und (text) positions stay vanilla.
        gen_mask = packed_sequence.new_zeros(packed_sequence.shape[0], dtype=torch.bool)
        gen_mask[packed_gen_token_indexes] = True

        biased_masks: list = []
        offset = 0
        for n, m in zip(sample_lens, attention_mask):
            sample_sal = full_saliency[offset : offset + n]
            sample_gen = gen_mask[offset : offset + n]
            offset += n
            if not sample_gen.any():
                biased_masks.append(m)
                continue
            bias = sbsr.bias(sample_sal)  # (n, n)
            # Zero out und<->und and und<->gen entries; SBSR only biases gen path.
            gen_pair = sample_gen.unsqueeze(-1) & sample_gen.unsqueeze(-2)
            bias = torch.where(gen_pair, bias, bias.new_zeros(()))
            bias = bias.to(m.dtype)
            # Broadcast to whatever shape `m` carries (e.g. (1, n, n), (H, n, n)).
            if m.dim() >= 2 and m.shape[-2:] == (n, n):
                view_shape = [1] * (m.dim() - 2) + [n, n]
                m_biased = m + bias.view(view_shape)
            else:
                raise RuntimeError(
                    f"SBSR: unexpected attention_mask shape {tuple(m.shape)}, "
                    f"expected last two dims = ({n}, {n})."
                )

            if sbsr.top_k is not None and sample_gen.sum().item() > sbsr.top_k:
                # Approximate top-k pre-softmax: keep the top-k bias columns
                # per gen-query row, push the rest to -inf via the additive mask.
                # We use the bias itself as the ranking key — since logits are
                # not visible at this hook site, biasing by `bias` is a valid
                # proxy under the SBSR assumption that saliency dominates.
                last_two = bias.view(-1, n, n)
                _, topk_idx = last_two.topk(sbsr.top_k, dim=-1)
                keep = torch.zeros_like(last_two, dtype=torch.bool)
                keep.scatter_(-1, topk_idx, True)
                # Only sparsify gen-query rows.
                gen_rows = sample_gen.unsqueeze(0).unsqueeze(-1)
                sparse = torch.where(
                    keep | ~gen_rows, last_two.new_zeros(()), last_two.new_full((), float("-inf"))
                )
                sparse = sparse.view_as(bias)
                m_biased = m_biased + sparse.view([1] * (m.dim() - 2) + [n, n])

            biased_masks.append(m_biased)

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

    root = _resolve_root(model)
    n = 0
    for _, module in root.named_modules():
        if isinstance(module, PackedAttentionMoT) and getattr(module, "_sbsr_patched", False):
            original = getattr(module, "_sbsr_original_forward_train", None)
            if original is not None:
                module.forward_train = original  # type: ignore[method-assign]
                delattr(module, "_sbsr_original_forward_train")
            else:
                # Fallback: drop the instance attribute and let class method
                # take over (only safe if class method was not itself replaced).
                if "forward_train" in module.__dict__:
                    del module.__dict__["forward_train"]
            module._sbsr_patched = False  # type: ignore[attr-defined]
            n += 1
    return n
