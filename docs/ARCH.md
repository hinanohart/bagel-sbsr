# Architecture

## Backbone integration strategy: `huggingface_hub` snapshot, runtime download

We **do not** vendor or git-submodule the BAGEL upstream repository. The reasoning:

1. BAGEL weights (~15 GB) cannot be committed.
2. BAGEL code (`ByteDance-Seed/Bagel`) evolves; a submodule pin tends to bit-rot.
3. Apache-2.0 lets us import classes directly; we only need to monkey-patch one
   attention method to install SBSR.

Approach:

- `scripts/download_bagel.py` calls `huggingface_hub.snapshot_download("ByteDance-Seed/BAGEL-7B-MoT")`
  using `HF_TOKEN` from the environment. Cached under `weights/bagel-7b-mot/`.
- The BAGEL Python source is fetched by adding `ByteDance-Seed/Bagel` as an optional
  install via a small `scripts/install_bagel_src.sh` (clones to `vendor/bagel-upstream/`,
  adds to `sys.path` at runtime).
- `src/bagel_sbsr/sbsr.py` defines the SBSR module and a `patch_bagel(model)` function
  that installs the attention hook on `PackedAttentionMoT.forward_train`.

## SBSR module — what it does

Given hidden states $H \in \mathbb{R}^{B \times T \times D}$ and a saliency map
$s \in \mathbb{R}^{B \times T}$ (default in v0.1.0: per-patch L2 magnitude of the
FLUX VAE latent; opt-in alternative: SigLIP2 CLS attention rollout, which requires
patching SigLIP to expose attention weights), SBSR adds an additive bias to the
attention logits computed inside BAGEL's generation expert:

$$
\ell_{ij} \mathrel{+}= \lambda (s_i + s_j) - \mu |s_i - s_j|
$$

with two learnable scalars $(\lambda, \mu) \in \mathbb{R}^2$. After the bias is added,
a top-$k$ sparse mask retains only the $k$ largest logits per query row before softmax.

This is implemented as a pre-softmax hook on `PackedAttentionMoT.forward_train`. The
modification is approximately 30 lines.

## Generation head

Unchanged from BAGEL: rectified flow on the FLUX VAE latent space (16 channels). The
distillation in Stage 2 keeps the rectified-flow architecture but replaces the loss with
either the iMF objective or the DMD2 objective; both are drop-in compatible.

## Why a Mixture-of-Transformers, why this MoT specifically

BAGEL's MoT has two experts (understanding, generation) sharing the same multi-head
attention computation; only Q/K/V/O projections are split per expert. This means SBSR
applied at the shared attention step affects *both* paths consistently — which is the
exact integration property we need for an object-prior bias that has to inform
generation without de-railing understanding.

## What is NOT included in v0.1

- Video. v0.2+ may add HunyuanVideo 1.5 (Apache-2.0) or Wan2.2 (Apache-2.0) as a second
  generation expert via co-upcycling, but this is out of v0.1 scope.
- Audio. Out of v0.1 scope.
- Any non-commercial backbone (Flux dev, Sana weights, SD 3.5, HunyuanDiT 1.0). The
  motivation for picking BAGEL specifically is full Apache-2.0 commercial use.
- **Inference-time SBSR.** The SBSR hook is applied on
  `PackedAttentionMoT.forward_train` only. `forward_inference`
  (`flash_attn_varlen_func`) is *not* patched in v0.1, so inference-time
  top-k speedups are a v0.2 target.
- **BAGEL forward adapter.** The `train_s1/s2/s3.py` real-train paths
  raise `NotImplementedError` from a documented adapter slot; the
  packed-sequence collator that bridges `(image, caption)` batches to
  `Bagel.forward(sequence_length=, packed_text_ids=, padded_latent=,
  packed_timesteps=, ...)` is the v0.1.1 follow-up. See
  `docs/TRAINING.md` §"BAGEL forward adapter".
