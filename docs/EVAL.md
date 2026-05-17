# Evaluation pin-board

This document records the exact upstream repos and commit pins used by
the BAGEL-SBSR evaluation pipeline. Versions are frozen here so the
numbers in the model card can be reproduced.

## Metrics and references

| Metric | Source | Pinned ref | Install |
|---|---|---|---|
| FID-50k | `GaParmar/clean-fid` | tagged `v0.1.35` | `uv pip install clean-fid==0.1.35` |
| CLIP-T (ViT-L-14) | `mlfoundations/open_clip` | release `v2.30.0` | already in `pyproject.toml` |
| GenEval | `djghosh13/geneval` | commit `c2a3e4d` (2024-10-15) | clone + `pip install -e .` |
| T2I-CompBench | `Karine-Huang/T2I-CompBench` | commit `5f7c9a0` (2024-03-22) | clone + follow upstream README |
| HPSv2.1 | `tgxs002/HPSv2` | release `v2.1` | `uv pip install hpsv2==2.1` |

## Reference image sets

| Set | Source | Files |
|---|---|---|
| COCO-30k val | `cocodataset.org` | 30,000 PNGs at 512² resampled |
| Parti prompts | `google-research/parti-prompts` | 1,632 prompts |
| GenEval prompts | bundled with GenEval | 553 prompts × 4 samples |
| T2I-CompBench prompts | bundled with T2I-CompBench | 6,000 prompts × 10 samples |

## Running the full eval

```bash
uv pip install -e .[eval]
uv pip install hpsv2==2.1
git clone --branch c2a3e4d https://github.com/djghosh13/geneval third_party/geneval
git clone --branch 5f7c9a0 https://github.com/Karine-Huang/T2I-CompBench third_party/t2i_compbench

uv run scripts/eval.py \
    --config configs/eval.yaml \
    --ckpt runs/s3/ckpt-00020000.safetensors \
    --metrics fid clip_t geneval t2i_compbench hpsv2
```

The output goes to `runs/eval/<run_name>/scores.json`.

## Reproducibility ±1%

Per the architecture spec (`project_oss-architecture-final-2026-05-17`
line 89), eval is considered reproducible at ±1% if:
- the same checkpoint hash (SHA-256, in `.safetensors.sha256` sidecar)
- the same upstream evaluator commit (this file)
- the same reference image hashes (`docs/REFERENCE_HASHES.md`, TODO)

If any of these drifts, document the new pin in this file and re-run.
