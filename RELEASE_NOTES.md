# BAGEL-SBSR v0.1.0.dev — Release Notes (draft)

This release ships the code basis for BAGEL-SBSR (Saliency-biased Sparse
Routing on BAGEL-7B-MoT, distilled with iMF + DMD2). The training stages
S1 / S2 / S3 are implemented as runnable scripts. **No model weights are
shipped in this version** — the pipeline must be trained from scratch on
4×H100 (~10 days wall-clock, ~$2,880) before checkpoints can be released.

## What ships in v0.1.0.dev

### Code
- `src/bagel_sbsr/` — Saliency-biased Sparse Routing core
  - `SBSR` module: additive bias `λ(s_i+s_j) - μ|s_i-s_j|` + top-k mask
  - `LatentMagnitudeProvider` — saliency from FLUX VAE latent magnitude
  - `attention_rollout` / `saliency_from_attentions` — SigLIP rollout (opt-in)
  - `patch_bagel` / `unpatch_bagel` — install/restore SBSR on
    `PackedAttentionMoT.forward_train`
  - `pipeline_bagel_sbsr.BagelSBSRPipeline` — Diffusers-shaped adapter
- `scripts/`
  - `download_bagel.py` — pulls `ByteDance-Seed/BAGEL-7B-MoT` via HF Hub
  - `install_bagel_src.sh` — clones BAGEL upstream (pinned rev)
  - `sanity_inference.py` — T2I + I2T smoke test via `InterleaveInferencer`
  - `train_s1.py` — S1 LoRA warm-up (gen-expert q/v + SBSR), bf16 + QLoRA-4bit
  - `train_s2.py` — S2 dual distillation (`--track imf` or `--track dmd2`)
  - `train_s3.py` — S3 integrated fine-tune (winner + SBSR merge)
  - `eval.py` — FID / CLIP-T / GenEval / T2I-CompBench / HPSv2.1
  - `coyo_dataloader.py` — streaming-only loader for COYO-700M
  - `launch_runpod.py` — RunPod dispatch (HF_TOKEN/RUNPOD_API_KEY env-only)
- `comfyui/nodes.py` — ComfyUI custom nodes (Loader + Sampler)
- `demo/app.py` — HF Space Gradio demo

### Tests
- 30 unit tests (smoke marker), all green on CPU
  - rollout shape + row-stochasticity (incl. discard_ratio path)
  - bias formula matches closed-form spec
  - top-k retains exactly k finite (with tied logits)
  - constant-row saliency falls back to 0.5 (not 0)
  - patch/unpatch cycle restores `forward_train` verbatim
  - `require_layers` guard raises if no PackedAttentionMoT is found
  - `LatentMagnitudeProvider` lifecycle (set / reset / neutral default)

## Scope of v0.1.0 (honest)

- **BAGEL forward adapter not implemented.** `train_s1/s2/s3.py` ship
  the SBSR hook, iMF/DMD2 loss kernels, COYO streaming dataloader,
  NaN-detect rollback, safetensors checkpoint, and the dry-run path.
  The packed-sequence collator that bridges `(image, caption)` batches
  to `Bagel.forward(sequence_length=, packed_text_ids=, padded_latent=,
  packed_timesteps=, ...)` is a clearly labeled adapter slot that
  raises `NotImplementedError` until v0.1.1. See `docs/TRAINING.md`
  §"BAGEL forward adapter".
- **Training path only.** SBSR is patched on
  `PackedAttentionMoT.forward_train`; `forward_inference` (which uses
  `flash_attn_varlen_func`) is *not* patched in v0.1. Inference-time top-k
  speedups are a v0.2 target — they require a separate score_mod / mask
  injection compatible with flash_attn_varlen.
- **Saliency source.** v0.1 uses VAE latent magnitude saliency by
  default (no extra forward pass, no SigLIP modification). A SigLIP
  rollout provider is shipped but requires manually patching SigLIP to
  expose attention weights. The choice is exposed in `configs/s*.yaml:
  sbsr.saliency_source`.
- **No checkpoints.** Apache-2.0 weight release requires either training
  end-to-end on 4×H100 for ~$2,880, or a community-pooled compute drop.

## Known limitations / caveats

- `forward_inference` SBSR path is deferred to v0.2 (see hook.py comment).
- COYO-700M URLs are externally hosted; mirror-fallback fires if >30% of
  the first 1k examples fail to download.
- GenEval and T2I-CompBench evaluators require manual installation from
  their upstream repos (pins recorded in `docs/EVAL.md`).
- kluster.ai code-review verification ran on best-effort basis (trial
  expired). Replaced with a 3-agent (architect + reviewer + critic) audit
  before every commit; findings are recorded in commit messages.

## Reproducibility

- Architecture spec frozen in `docs/ARCH.md` and
  `project_oss-architecture-final-2026-05-17` memory (immutable).
- Training hyperparameters in `configs/s1.yaml`, `configs/s2.yaml`,
  `configs/s3.yaml` — no in-code constants.
- `--dry-run` flag on every train script exercises wire-up without
  weights or GPU (CI-friendly).
- Checkpoints are `.safetensors` only; pickle is avoided everywhere.

## License

Apache-2.0 (this repo). BAGEL-7B-MoT upstream weights are Apache-2.0
(ByteDance-Seed). COYO-700M is CC-BY-4.0 (Kakao Brain). See `docs/DATA.md`
for the full attribution chain and inappropriate-content review.

## Citing

```bibtex
@software{bagel_sbsr_2026,
  author       = {bagel-sbsr contributors},
  title        = {BAGEL-SBSR: Saliency-biased Sparse Routing on BAGEL-7B-MoT},
  year         = {2026},
  url          = {https://github.com/hinanohart/bagel-sbsr},
  version      = {0.1.0.dev0}
}
```
