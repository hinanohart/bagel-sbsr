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
   reusing the SigLIP2 vision tower's CLS attention rollout — no extra encoder.
2. **iMF + DMD2 dual distillation** — does an [improved Mean Flow](https://arxiv.org/abs/2512.02012)
   teacher beat [DMD2](https://tianweiy.github.io/dmd2/) for distilling a MoT generation
   expert down to 2-4 NFE? We run both, gate by FID/CLIP at the 10k-step mark, and ship
   the winner.

The repository is structured so each piece is independently usable: SBSR is a thin
attention hook that works on top of stock BAGEL, and the distillation pipeline accepts any
flow-matching student.

## Status

This is `v0.1.0.dev` — pre-release skeleton. See `docs/STATUS.md` for the rolling
phase tracker (P0 init → P8 release).

## Quick start (inference, after release)

```bash
uv pip install bagel-sbsr
python -m bagel_sbsr.demo --prompt "a photograph of an astronaut riding a horse" --nfe 4
```

## Repository layout

```
src/bagel_sbsr/        # SBSR module, pipeline, distillation losses
scripts/               # train_s1/s2/s3, eval, download, launch_runpod
tests/                 # unit + smoke tests
comfyui/               # ComfyUI custom node
demo/                  # Gradio HF Space app
papers/                # preprint draft (LaTeX)
experiments/_wip/      # failure museum (R8)
```

## Training recipe (summary)

| Stage | Trainable                                       | Data                       | Cost (4×H100) |
| ----- | ----------------------------------------------- | -------------------------- | ------------- |
| S1    | LoRA r=16 on gen-expert q/v + SBSR (λ, μ)       | COYO-700M (~50M subset)    | ~60h          |
| S2    | gen-expert full bf16 FT (iMF + DMD2 dual track) | S1 + teacher BAGEL@50-NFE  | ~140h         |
| S3    | gen-expert + SBSR merged                        | + GenEval/T2I-CompBench    | ~40h          |

Total: ~10 days on 4×H100, ~$2,880 at RunPod spot rates. See `docs/TRAINING.md`.

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
- [iMeanFlow](https://github.com/Lyy-iiis/imeanflow) for the MIT reference implementation
- [DMD2](https://tianweiy.github.io/dmd2/) for the distillation recipe
