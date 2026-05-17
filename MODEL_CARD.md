# Model Card: BAGEL-SBSR v0.1.0.dev

## Model description

BAGEL-SBSR is a unified multimodal generation/understanding model built on
ByteDance-Seed's BAGEL-7B-MoT (a 2-expert Mixture-of-Transformers with
shared attention and rectified-flow generation on FLUX VAE latents). Two
modifications are added:

1. **SBSR (Saliency-biased Sparse Routing)** — an additive attention bias
   `λ(s_i + s_j) - μ|s_i - s_j|` followed by top-k sparsification, where
   `s_i` is a per-patch saliency derived from the FLUX VAE latent
   magnitude. Two learnable scalars (λ, μ) are trained jointly.
2. **iMF / DMD2 dual distillation** — at Stage 2, two distillation losses
   (improved Mean Flow and Distribution Matching Distillation v2) are
   trained in parallel; a FID/CLIP gate at 10k steps decides which to
   continue with.

## Intended use

- Text-to-image generation
- Image-to-text understanding (captioning, VQA)
- Research on saliency-conditioned sparse routing in MoT models

## Out-of-scope

- Real-time video generation (deferred to v0.2+)
- Audio
- Production deployment without further safety alignment (no RLHF /
  preference-aligned tuning in v0.1)

## Training data

- **S1**: COYO-700M (CC-BY-4.0, Kakao Brain) — 50M-image subset, streaming
  only. URL liveness ≥ 70% required at step 0; if it drops below
  threshold during the first 1k examples, training aborts with a
  mirror-fallback hint.
- **S2 / S3**: COYO + GenEval/T2I-CompBench train splits (see configs).

We deliberately avoid LAION-5B and its derivatives due to the 2023 CSAM
review and the subsequent dataset withdrawal.

## Performance

This release contains *no weights* — performance numbers will be added
when a checkpoint is published. Target headlines (from architecture spec):

- FID-50k on COCO-val: target < 30 at 4 NFE
- CLIP-T on Parti: target > 0.30 at 4 NFE
- T2I-CompBench Attribute / Object: target ≥ BAGEL-7B-MoT baseline

## Compute and emissions

Estimated training cost (4×H100 SXM at $1.30-1.60/h spot):
- S1 LoRA warm-up: ~60h / ~$720
- S2 distillation: ~140h / ~$1,680
- S3 integration: ~40h / ~$480
- **Total: ~240h / ~$2,880**

## Limitations and risks

- **Hallucinations**: as with all multimodal generators, fabricated
  details are possible. Do not use for medical / legal / financial advice.
- **Bias**: training on web-sourced COYO inherits the dataset's biases.
- **Saliency source caveat**: VAE latent magnitude is a *weak* proxy for
  semantic saliency. It correlates with image-content energy rather than
  human attention. The SigLIP rollout provider (opt-in) is closer to the
  original SBSR design intent but requires patching SigLIP.

## License

- Code: Apache-2.0
- BAGEL-7B-MoT base: Apache-2.0 (ByteDance-Seed)
- COYO-700M: CC-BY-4.0 (Kakao Brain)
- iMF reference impl (Lyy-iiis/imeanflow): MIT — compatible with Apache-2.0

## Citation

See `RELEASE_NOTES.md` for the BibTeX entry. BAGEL upstream:
`arXiv:2505.14683`. iMF: `arXiv:2512.02012`. DMD2: tianweiy.github.io/dmd2.

## Contact

Open an issue at https://github.com/hinanohart/bagel-sbsr/issues.
