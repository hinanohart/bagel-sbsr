"""Unit tests for SBSR core: saliency rollout, bias formula, top-k masking, hook lifecycle."""

from __future__ import annotations

import math
import sys
import types

import pytest
import torch


@pytest.mark.smoke
def test_sbsr_imports():
    from bagel_sbsr import (
        SBSR,
        LatentMagnitudeProvider,
        attention_rollout,
        latent_magnitude_saliency,
        saliency_from_attentions,
    )

    assert callable(attention_rollout)
    assert callable(saliency_from_attentions)
    assert callable(latent_magnitude_saliency)
    assert LatentMagnitudeProvider is not None
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
def test_attention_rollout_discard_ratio_preserves_per_row_sparsity():
    from bagel_sbsr.saliency import attention_rollout

    B, H, T = 1, 1, 6
    a = torch.softmax(torch.randn(B, H, T, T), dim=-1)
    rolled = attention_rollout([a], discard_ratio=0.5)
    # Each row should still sum to 1 (renormalisation invariant).
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
def test_saliency_constant_row_falls_back_to_neutral():
    from bagel_sbsr.saliency import saliency_from_attentions

    # Construct an attention where the rollout will produce a uniform CLS row.
    B, H, T = 1, 1, 4
    attn = torch.full((B, H, T, T), 1.0 / T)
    sal = saliency_from_attentions([attn], cls_index=0, normalize=True)
    # min == max, so the fallback path returns 0.5.
    assert torch.allclose(sal, torch.full_like(sal, 0.5))


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
def test_top_k_mask_keeps_exactly_k_with_ties():
    """Regression: previous `>=` impl produced > k finite when ties existed."""
    from bagel_sbsr.sbsr import SBSR

    torch.manual_seed(0)
    sbsr = SBSR(top_k=3, learnable=False)
    logits = torch.randn(2, 5, 8)
    # Inject ties: copy column 0 into column 5 and 6.
    logits[..., 5] = logits[..., 0]
    logits[..., 6] = logits[..., 0]
    out = sbsr.apply_top_k(logits)

    finite_count_per_row = torch.isfinite(out).sum(dim=-1)
    assert (finite_count_per_row == 3).all(), finite_count_per_row


@pytest.mark.smoke
def test_top_k_mask_method_returns_additive_form():
    from bagel_sbsr.sbsr import SBSR

    sbsr = SBSR(top_k=2, learnable=False)
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    mask = sbsr.top_k_mask(logits)
    # exactly 2 zeros and 2 -inf per row
    assert (mask == 0).sum() == 2
    assert torch.isinf(mask).sum() == 2


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
def test_latent_magnitude_saliency_shape_and_range():
    from bagel_sbsr import latent_magnitude_saliency

    latent = torch.randn(2, 16, 8, 8)
    sal = latent_magnitude_saliency(latent)
    assert sal.shape == (2, 64)
    assert sal.min() >= 0.0
    assert sal.max() <= 1.0


@pytest.mark.smoke
def test_latent_magnitude_provider_neutral_until_set():
    from bagel_sbsr import LatentMagnitudeProvider

    provider = LatentMagnitudeProvider()
    packed = torch.zeros(10, 768)
    idx = torch.arange(10)
    sal = provider(packed, idx)
    assert sal.shape == (10,)
    assert torch.allclose(sal, torch.full_like(sal, 0.5))


@pytest.mark.smoke
def test_latent_magnitude_provider_reset():
    from bagel_sbsr import LatentMagnitudeProvider

    provider = LatentMagnitudeProvider()
    provider.set_latent(torch.randn(1, 16, 4, 4))
    assert provider._saliency is not None
    provider.reset()
    assert provider._saliency is None


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

    if "modeling.bagel.qwen2_navit" in sys.modules:
        pytest.skip(
            "BAGEL upstream is importable; this test exercises the missing-vendor path only"
        )

    with pytest.raises(BagelNotInstalledError):
        patch_bagel(DummyModel())


def _install_fake_bagel_module(monkeypatch):
    """Install a fake `modeling.bagel.qwen2_navit.PackedAttentionMoT` so the
    hook code path can be exercised on CPU without the real BAGEL upstream."""

    class FakePackedAttentionMoT(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward_train(
            self,
            packed_sequence,
            sample_lens,
            attention_mask,
            packed_position_embeddings,
            packed_und_token_indexes,
            packed_gen_token_indexes,
        ):
            # Echo back the (possibly biased) attention_mask for assertions.
            return ("ok", attention_mask)

    pkg_modeling = types.ModuleType("modeling")
    pkg_bagel = types.ModuleType("modeling.bagel")
    pkg_qwen = types.ModuleType("modeling.bagel.qwen2_navit")
    pkg_qwen.PackedAttentionMoT = FakePackedAttentionMoT
    pkg_modeling.bagel = pkg_bagel
    pkg_bagel.qwen2_navit = pkg_qwen
    monkeypatch.setitem(sys.modules, "modeling", pkg_modeling)
    monkeypatch.setitem(sys.modules, "modeling.bagel", pkg_bagel)
    monkeypatch.setitem(sys.modules, "modeling.bagel.qwen2_navit", pkg_qwen)
    return FakePackedAttentionMoT


@pytest.mark.smoke
def test_patch_and_unpatch_cycle_with_fake_bagel(monkeypatch):
    """Wire-up sanity: patch_bagel installs SBSR, calls forward_train through
    the wrapper with bias added, then unpatch_bagel restores the original."""
    FakeAttn = _install_fake_bagel_module(monkeypatch)
    from bagel_sbsr import LatentMagnitudeProvider, patch_bagel, unpatch_bagel

    class FakeBagel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn0 = FakeAttn()
            self.attn1 = FakeAttn()

    model = FakeBagel()
    provider = LatentMagnitudeProvider()
    provider.set_latent(torch.randn(1, 16, 2, 2))  # 4 patches

    patched = patch_bagel(
        model,
        lambda_init=1.0,
        mu_init=0.0,
        top_k=None,
        saliency_provider=provider,
        learnable=True,
        require_layers=2,
    )
    assert len(patched) == 2

    # Call the wrapped forward_train with a 4-token packed sequence,
    # gen indices = [2, 3], und indices = [0, 1].
    n = 4
    packed_sequence = torch.zeros(n, 8)
    attention_mask = [torch.zeros(1, n, n)]
    sample_lens = [n]
    packed_und = torch.tensor([0, 1], dtype=torch.long)
    packed_gen = torch.tensor([2, 3], dtype=torch.long)
    pos = torch.zeros(n, 8)

    _, biased = model.attn0.forward_train(
        packed_sequence, sample_lens, attention_mask, pos, packed_und, packed_gen
    )
    # The gen<->gen 2x2 block should differ from zero; und positions should not.
    bm = biased[0][0]  # (n, n)
    gen_block = bm[2:4, 2:4]
    und_und_block = bm[0:2, 0:2]
    assert (gen_block != 0).any()
    assert torch.equal(und_und_block, torch.zeros_like(und_und_block))

    # Lambda parameter should accumulate grad from wrapped path via SBSR.
    sbsr = model.sbsr
    loss = bm.sum()
    loss.backward()
    assert sbsr.lambda_.grad is not None

    # Unpatch restores the original method.
    restored = unpatch_bagel(model)
    assert restored == 2
    # After unpatch, attention_mask comes back unchanged (no bias).
    _, biased2 = model.attn0.forward_train(
        packed_sequence, sample_lens, [torch.zeros(1, n, n)], pos, packed_und, packed_gen
    )
    assert torch.equal(biased2[0], torch.zeros_like(biased2[0]))


@pytest.mark.smoke
def test_patch_bagel_require_layers_raises_when_zero(monkeypatch):
    _install_fake_bagel_module(monkeypatch)
    from bagel_sbsr import patch_bagel

    class Empty(torch.nn.Module):
        pass

    with pytest.raises(ValueError, match="require_layers"):
        patch_bagel(Empty())
