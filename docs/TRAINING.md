# Training recipe

Three-stage curriculum. All stages assume the BAGEL-7B-MoT backbone has been downloaded
via `scripts/download_bagel.py` (uses `HF_TOKEN` from environment).

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

- ~240h on 4×H100, total ~$2,880 at RunPod H100-SXM spot ($2.79-3.49/h)
- Alternative: ~130h on 8×H100, total ~$2,879
- Budget gate: $4,000 hard ceiling; auto-pause and notify user above this

## Ablation grid

`(λ, μ) ∈ {0.0, 0.5, 1.0}² × top-k ∈ {32, 64, 128} × NFE ∈ {1, 2, 4}` plus 8 named cells
documented in `docs/ABLATION.md` (created in P6).
