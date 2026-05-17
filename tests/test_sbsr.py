"""Unit tests for SBSR core: saliency rollout, bias formula, top-k masking."""

from __future__ import annotations

import math

import pytest
import torch


@pytest.mark.smoke
def test_sbsr_imports():
    from bagel_sbsr.saliency import attention_rollout, saliency_from_attentions
    from bagel_sbsr.sbsr import SBSR

    assert callable(attention_rollout)
    assert callable(saliency_from_attentions)
    assert SBSR is not None


@pytest.mark.smoke
def test_attention_rollout_shape_and_normalisation():
    from bagel_sbsr.saliency import attention_rollout

    B, H, T, L = 2, 4, 5, 3
    attentions = [torch.softmax(torch.randn(B, H, T, T), dim=-1) for _ in range(L)]
    rolled = attention_rollout(attentions, head_fusion="mean")

    assert rolled.shape == (B, T, T)
    row_sums = rolled.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


@pytest.mark.smoke
def test_saliency_extracts_non_cls_tokens():
    from bagel_sbsr.saliency import saliency_from_attentions

    B, H, T, L = 1, 2, 6, 2
    attentions = [torch.softmax(torch.randn(B, H, T, T), dim=-1) for _ in range(L)]
    sal = saliency_from_attentions(attentions, cls_index=0, normalize=True)

    assert sal.shape == (B, T - 1)
    assert sal.min() >= 0.0
    assert sal.max() <= 1.0


@pytest.mark.smoke
def test_sbsr_bias_formula_matches_spec():
    from bagel_sbsr.sbsr import SBSR

    sbsr = SBSR(lambda_init=1.0, mu_init=0.5, top_k=None, learnable=False)
    s = torch.tensor([0.2, 0.8, 0.5])
    bias = sbsr.bias(s)

    expected = torch.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            expected[i, j] = 1.0 * (s[i] + s[j]) - 0.5 * abs(s[i] - s[j])

    assert bias.shape == (3, 3)
    assert torch.allclose(bias, expected, atol=1e-6)


@pytest.mark.smoke
def test_sbsr_bias_batched():
    from bagel_sbsr.sbsr import SBSR

    sbsr = SBSR(lambda_init=0.5, mu_init=0.25, top_k=None, learnable=False)
    s = torch.rand(2, 4)
    bias = sbsr.bias(s)
    assert bias.shape == (2, 4, 4)


@pytest.mark.smoke
def test_top_k_mask_keeps_exactly_k():
    from bagel_sbsr.sbsr import SBSR

    torch.manual_seed(0)
    sbsr = SBSR(top_k=3, learnable=False)
    logits = torch.randn(2, 5, 8)
    out = sbsr.apply_top_k(logits)

    finite_count_per_row = torch.isfinite(out).sum(dim=-1)
    assert (finite_count_per_row == 3).all(), finite_count_per_row


@pytest.mark.smoke
def test_top_k_none_is_passthrough():
    from bagel_sbsr.sbsr import SBSR

    sbsr = SBSR(top_k=None, learnable=False)
    logits = torch.randn(3, 7)
    out = sbsr.apply_top_k(logits)
    assert torch.equal(out, logits)


@pytest.mark.smoke
def test_top_k_larger_than_T_is_passthrough():
    from bagel_sbsr.sbsr import SBSR

    sbsr = SBSR(top_k=100, learnable=False)
    logits = torch.randn(2, 4)
    out = sbsr.apply_top_k(logits)
    assert torch.equal(out, logits)


@pytest.mark.smoke
def test_sbsr_parameters_are_learnable_when_requested():
    from bagel_sbsr.sbsr import SBSR

    sbsr = SBSR(lambda_init=0.5, mu_init=0.25, learnable=True)
    params = list(sbsr.parameters())
    assert len(params) == 2
    for p in params:
        assert p.requires_grad


@pytest.mark.smoke
def test_sbsr_parameters_are_frozen_when_requested():
    from bagel_sbsr.sbsr import SBSR

    sbsr = SBSR(lambda_init=0.5, mu_init=0.25, learnable=False)
    assert list(sbsr.parameters()) == []


@pytest.mark.smoke
def test_sbsr_gradient_flows_to_lambda_mu():
    from bagel_sbsr.sbsr import SBSR

    sbsr = SBSR(lambda_init=0.5, mu_init=0.25, top_k=None, learnable=True)
    s = torch.tensor([0.1, 0.9, 0.5, 0.3])
    logits = torch.zeros(4, 4, requires_grad=False)
    out = sbsr(logits, s, apply_top_k=False)
    loss = out.sum()
    loss.backward()

    assert sbsr.lambda_.grad is not None
    assert sbsr.mu_.grad is not None
    assert not math.isnan(sbsr.lambda_.grad.item())
    assert not math.isnan(sbsr.mu_.grad.item())


@pytest.mark.smoke
def test_sbsr_forward_combines_bias_and_topk():
    from bagel_sbsr.sbsr import SBSR

    sbsr = SBSR(lambda_init=1.0, mu_init=0.0, top_k=2, learnable=False)
    s = torch.tensor([0.1, 0.9, 0.5, 0.3])
    logits = torch.zeros(4, 4)
    out = sbsr(logits, s, apply_top_k=True)

    finite_count = torch.isfinite(out).sum(dim=-1)
    assert (finite_count == 2).all()


@pytest.mark.smoke
def test_hook_module_import_does_not_require_bagel():
    from bagel_sbsr import hook

    assert hasattr(hook, "patch_bagel")
    assert hasattr(hook, "BagelNotInstalledError")


@pytest.mark.smoke
def test_patch_bagel_raises_clean_error_when_vendor_missing():
    from bagel_sbsr.hook import BagelNotInstalledError, patch_bagel

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

    if "modeling.bagel.qwen2_navit" in __import__("sys").modules:
        pytest.skip(
            "BAGEL upstream is importable; this test exercises the missing-vendor path only"
        )

    with pytest.raises(BagelNotInstalledError):
        patch_bagel(DummyModel())
