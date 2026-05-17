# Training recipe

Three-stage curriculum. All stages assume the BAGEL-7B-MoT backbone has been downloaded
via `scripts/download_bagel.py` (uses `HF_TOKEN` from environment).

## BAGEL forward adapter (v0.1.1 follow-up)

`scripts/train_s1.py`, `train_s2.py`, and `train_s3.py` ship the SBSR
hook, the iMF/DMD2 loss kernels, the COYO streaming dataloader, the
NaN-detect rollback, and the safetensors checkpoint manifest. What is
**not** shipped in v0.1.0.dev is the packed-sequence collator that
bridges a `(image, caption)` batch to the upstream `Bagel.forward`
signature:

```python
model(
    sequence_length=N,
    packed_text_ids=...,            # 1-D long, packed tokenizer output
    packed_text_indexes=...,        # 1-D long, where text tokens land in seq
    sample_lens=[n_i for i in B],
    packed_position_ids=...,
    padded_latent=...,              # FLUX VAE latent per sample
    patchified_vae_latent_shapes=[(h, w), ...],
    packed_latent_position_ids=...,
    packed_vae_token_indexes=...,
    packed_timesteps=...,           # 1-D float, flow-matching t
    mse_loss_indexes=...,           # bool mask for MSE-loss positions
    ce_loss_indexes=...,            # bool mask for CE-loss positions
    packed_label_ids=...,
)
```
(see `vendor/bagel-upstream/modeling/bagel/bagel.py:101`).

Both `train_s1._bagel_forward_adapter` and
`train_s2._bagel_velocity_adapter` raise `NotImplementedError` from a
clearly labeled adapter slot. The adapter is intentionally not written
without GPU-side iteration: shape/dtype mismatches in the collator are
the kind of bug that needs a real BAGEL forward pass to surface, which
this development environment does not have.

## Stage 1 — SBSR warm-up (LoRA)

| Setting       | Value                                                                       |
| ------------- | --------------------------------------------------------------------------- |
| Trainable     | LoRA rank=16 on gen-expert q/v projections, plus SBSR scalars (λ, μ)        |
| Frozen        | Everything else (understanding expert, embeddings, shared attention norms)  |
| Precision     | bf16 forward; QLoRA-4bit storage permitted                                  |
| Data          | COYO-700M, ~50M-sample filtered subset (CC-BY-4.0)                          |
| Batch size    | 256 (global) — 64 per H100 × 4                                              |
| Learning rate | 1e-4, cosine warmup 1k steps → cosine decay                                 |
| Steps         | 30k                                                                         |
| Wall clock    | ~60h on 4×H100                                                              |
| Cost          | ~$720 at RunPod H100-SXM spot                                               |

Run: `uv run scripts/train_s1.py --config configs/s1.yaml`

## Stage 2 — Distillation (iMF + DMD2 dual track)

| Setting       | Value                                                                                          |
| ------------- | ---------------------------------------------------------------------------------------------- |
| Trainable     | gen-expert (full, bf16) — **QLoRA is NOT used here** (4-bit quantization erases velocity signal) |
| Loss          | Two parallel runs: iMF (Alg. 1 of arXiv:2512.02012) and DMD2 (arXiv:2405.14867)                |
| Selection     | At 10k steps, compare COCO-2017 FID@4-NFE + CLIP-T@4-NFE. Keep winner, drop loser.             |
| Teacher       | BAGEL@50-NFE sampled offline                                                                   |
| Precision     | bf16 throughout                                                                                |
| Batch size    | 128 (global)                                                                                   |
| Learning rate | 1e-5 (EMF arXiv:2604.18168 recipe)                                                             |
| Steps         | 80k                                                                                            |
| Wall clock    | ~140h on 4×H100                                                                                |
| Cost          | ~$1,680                                                                                        |

Run: `uv run scripts/train_s2.py --config configs/s2.yaml`

## Stage 3 — Integration fine-tune

| Setting       | Value                                                                |
| ------------- | -------------------------------------------------------------------- |
| Trainable     | gen-expert + SBSR (merged, no LoRA)                                  |
| Data          | S1/S2 mix + GenEval train + T2I-CompBench train                      |
| Precision     | bf16                                                                 |
| Batch size    | 128                                                                  |
| Learning rate | 5e-6                                                                 |
| Steps         | 20k                                                                  |
| Wall clock    | ~40h on 4×H100                                                       |
| Cost          | ~$480                                                                |

Run: `uv run scripts/train_s3.py --config configs/s3.yaml`

## Totals

Wall-clock: ~240h on 4×H100. Total spend depends on the GPU pool:

- H100 SXM on-demand (~$2.69-2.99/h × 4 × 240h): **~$2,580-$2,870**
- H100 SXM spot when available: roughly half
- H100 PCIe on-demand (~$1.99/h × 4 × 240h): **~$1,910**

Budget gate: `safety.budget_hard_ceiling_usd` (default $4,000) is
enforced by `scripts/launch_runpod.py` — the poll loop calls
`runpod.terminate_pod` when accumulated cost exceeds the per-stage cap
(`cluster.max_cost_usd`, clamped to the hard ceiling).

## Ablation grid

`(λ, μ) ∈ {0.0, 0.5, 1.0}² × top-k ∈ {32, 64, 128} × NFE ∈ {1, 2, 4}` plus 8 named cells
documented in `docs/ABLATION.md` (created in P6).
