# BAGEL-SBSR

> **Saliency-biased Sparse Routing for Mixture-of-Transformers, with iMF/DMD2 dual distillation.**
> A drop-in extension of [BAGEL-7B-MoT](https://github.com/ByteDance-Seed/Bagel) that adds an
> object-prior attention routing bias and a 2-4 NFE generation pipeline, fully Apache-2.0.

## What this is

BAGEL-SBSR is a research OSS that explores two questions on top of a strong unified
multimodal backbone (BAGEL-7B, Apache-2.0, Mixture-of-Transformers with shared attention):

1. **Saliency-biased Sparse Routing (SBSR)** — can ~30 lines of attention modification
   (additive logit bias from a saliency map, plus top-k sparsification) measurably improve
   compositional T2I scores at near-zero added cost? Saliency is computed for free by
   reusing the FLUX VAE latent magnitude (no extra forward pass). A SigLIP rollout
   provider is also shipped as an opt-in alternative for users who patch SigLIP to
   expose attention weights.
2. **iMF + DMD2 dual distillation** — does an [improved Mean Flow](https://arxiv.org/abs/2512.02012)
   teacher beat [DMD2](https://tianweiy.github.io/dmd2/) for distilling a MoT generation
   expert down to 2-4 NFE? We run both, gate by FID/CLIP at the 10k-step mark, and ship
   the winner.

The repository is structured so each piece is independently usable: SBSR is a thin
attention hook that works on top of stock BAGEL, and the distillation pipeline accepts any
flow-matching student.

## Status

This is `v0.1.0.dev` — pre-release. All code (P0 → P8) is committed:
SBSR core, S1/S2/S3 training entrypoints, evaluation pipeline,
ComfyUI nodes, Diffusers pipeline adapter, HF Space demo, model card,
release notes, and preprint outline. **No model weights** are shipped
in v0.1.0.dev; training requires the steps in [MANUAL.md](MANUAL.md)
(~10 days on 4×H100, ~\$2,880 spot).

See `docs/STATUS.md` for the per-phase tracker.

## Quick start (no GPU, no secrets)

```bash
git clone https://github.com/hinanohart/bagel-sbsr.git && cd bagel-sbsr
uv sync
uv run pytest -q -m smoke                                          # 30 passed
uv run scripts/train_s1.py --config configs/s1.yaml --dry-run      # patch/unpatch cycle
uv run scripts/train_s2.py --config configs/s2.yaml --track imf --dry-run
uv run scripts/train_s3.py --config configs/s3.yaml --dry-run
uv run scripts/eval.py    --config configs/eval.yaml --dry-run
```

Each dry-run exercises wire-up + numerical kernels on synthetic data so
issues surface before you touch a GPU.

## Repository layout

```
src/bagel_sbsr/        # SBSR core (sbsr, hook, saliency, latent_saliency, pipeline)
scripts/               # train_s1/s2/s3, eval, coyo_dataloader, launch_runpod, launch_full.sh
tests/                 # 30 smoke tests (CPU-only)
comfyui/               # BagelSBSRLoader + BagelSBSRSampler nodes
demo/                  # Gradio HF Space app
papers/                # preprint outline (LaTeX)
docs/                  # STATUS, TRAINING, DATA, ARCH, EVAL
configs/               # s1.yaml, s2.yaml, s3.yaml, eval.yaml
experiments/_wip/      # R8 failure museum
```

## Training recipe (summary)

| Stage | Trainable                                       | Data                       | Cost (4×H100) |
| ----- | ----------------------------------------------- | -------------------------- | ------------- |
| S1    | LoRA r=16 on gen-expert q/v + SBSR (λ, μ)       | COYO-700M (~50M subset)    | ~60h          |
| S2    | gen-expert full bf16 FT (iMF + DMD2 dual track) | S1 + teacher BAGEL@50-NFE  | ~140h         |
| S3    | gen-expert + SBSR merged                        | + GenEval/T2I-CompBench    | ~40h          |

Total: ~10 days on 4×H100, ~\$2,880 at RunPod spot rates. See `docs/TRAINING.md`.

**Scope of v0.1.0 (honest)**: SBSR is installed on
`PackedAttentionMoT.forward_train` only. The `forward_inference` path
(which uses `flash_attn_varlen_func`) is *not* patched in v0.1, so
inference-time top-k speedups are a v0.2 target. See `RELEASE_NOTES.md`.

## Launching training (user prerequisites)

See [MANUAL.md](MANUAL.md) for the irreducible manual steps (setting
HF_TOKEN / RUNPOD_API_KEY, running `scripts/launch_full.sh`). Tokens
are read by scripts from env vars only — they never appear in
process listings or logs.

## License

- **Code & weights** (this repo): Apache License 2.0
- **Backbone**: BAGEL-7B-MoT by ByteDance-Seed, Apache-2.0
- **Training data**: COYO-700M (CC-BY-4.0) + JourneyDB (research-only, not redistributed)

LAION-aesthetic v2 is intentionally **excluded** (see `docs/DATA.md` §Safety).

## Citation

See `CITATION.cff`. Preprint forthcoming with v0.1.0 release.

## Discussion: framing notes

The engineering framing of SBSR is straightforward: an object-prior bias on attention
logits, drawn from a precomputed saliency map. The motivation overlaps with
[Gestalt figure/ground organization](https://en.wikipedia.org/wiki/Figure%E2%80%93ground_(perception))
in vision science, and—more loosely—Marshall McLuhan's "figure-ground" reading of media
([Laws of Media](https://en.wikipedia.org/wiki/Laws_of_Media), 1988). We mention this only
to credit the conceptual lineage; the empirical claims in this repo stand or fall on
GenEval / T2I-CompBench / HPSv2.1, not on theoretical appeals. See `papers/preprint.tex`
§Discussion for one paragraph on this.

## Contributing

Issues and PRs welcome. Please run `ruff check` and `pytest -m smoke` before submitting.

## Acknowledgements

- [BAGEL](https://github.com/ByteDance-Seed/Bagel) team for the Apache-2.0 MoT backbone
- [iMeanFlow](https://github.com/Lyy-iiis/imeanflow) for the MIT-licensed reference implementation (Apache-2.0 compatible)
- [DMD2](https://tianweiy.github.io/dmd2/) for the distillation recipe
