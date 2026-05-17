"""Saliency-biased Sparse Routing (SBSR) — bias computation and top-k masking.

Given a per-token saliency $s \\in [0, 1]^{B \\times T}$, SBSR adds an additive
bias to the attention logits

    b_{ij} = lambda * (s_i + s_j) - mu * |s_i - s_j|

with two learnable scalars (lambda, mu). After the bias is added, a top-k
sparse mask retains only the k largest logits per query row before softmax,
sparsifying the routing in a saliency-aware fashion.

The module is BAGEL-agnostic; it consumes saliency and logits and produces a
modified logits tensor. Integration with BAGEL's `PackedAttentionMoT.forward_train`
is in `hook.py`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SBSR"]


class SBSR(nn.Module):
    """Saliency-biased Sparse Routing module.

    Args:
        lambda_init: initial value of the additive saliency-sum coefficient.
        mu_init: initial value of the saliency-difference coefficient.
        top_k: optional sparsification. If set and < T, retains only the top-k
            logits per query row. If None, sparsification is disabled.
        learnable: if False, lambda and mu are buffers (not trained). Default True.
    """

    def __init__(
        self,
        lambda_init: float = 0.5,
        mu_init: float = 0.25,
        top_k: int | None = 64,
        *,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if learnable:
            self.lambda_ = nn.Parameter(torch.tensor(float(lambda_init)))
            self.mu_ = nn.Parameter(torch.tensor(float(mu_init)))
        else:
            self.register_buffer("lambda_", torch.tensor(float(lambda_init)))
            self.register_buffer("mu_", torch.tensor(float(mu_init)))
        self.top_k = top_k

    def bias(self, saliency: torch.Tensor) -> torch.Tensor:
        """Compute the SBSR additive bias from saliency.

        Args:
            saliency: (B, T) or (T,) saliency tensor.

        Returns:
            bias of shape (B, T, T) or (T, T) matching input batch dim.
        """
        if saliency.dim() == 1 or saliency.dim() == 2:
            s_i = saliency.unsqueeze(-1)
            s_j = saliency.unsqueeze(-2)
        else:
            raise ValueError(f"saliency must be 1D or 2D, got shape {saliency.shape}")

        return self.lambda_ * (s_i + s_j) - self.mu_ * (s_i - s_j).abs()

    def apply_top_k(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply top-k sparsification along the last dim.

        Args:
            logits: (..., T_q, T_k) logits.

        Returns:
            logits with positions outside the top-k per row replaced by -inf.
        """
        if self.top_k is None:
            return logits
        t_k = logits.shape[-1]
        if self.top_k >= t_k:
            return logits

        topk_vals, _ = logits.topk(self.top_k, dim=-1)
        threshold = topk_vals[..., -1:].detach()
        return torch.where(logits >= threshold, logits, logits.new_full((), float("-inf")))

    def forward(
        self,
        logits: torch.Tensor,
        saliency: torch.Tensor,
        *,
        apply_top_k: bool = True,
    ) -> torch.Tensor:
        """Combined bias + (optional) top-k mask on attention logits.

        Args:
            logits: (..., T_q, T_k) or (..., T, T) — attention logits before softmax.
            saliency: (T,) or (B, T) saliency, must broadcast against logits.
            apply_top_k: whether to sparsify after biasing.
        """
        bias = self.bias(saliency)
        while bias.dim() < logits.dim():
            bias = bias.unsqueeze(-3)
        out = logits + bias.to(logits.dtype)
        if apply_top_k:
            out = self.apply_top_k(out)
        return out

    def extra_repr(self) -> str:
        return f"lambda_init={float(self.lambda_):.3f}, mu_init={float(self.mu_):.3f}, top_k={self.top_k}"
