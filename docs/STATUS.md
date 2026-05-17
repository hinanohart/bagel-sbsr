# Rolling phase tracker

This file tracks the autonomous build pipeline (P0 → P8). Updated at the end of each phase.

| Phase | Status        | Artifact                                                            | Notes                             |
| ----- | ------------- | ------------------------------------------------------------------- | --------------------------------- |
| P0    | ✅ completed  | repo init / LICENSE / pyproject / CI / docs stubs                   | scaffold + watcher fixes          |
| P1    | ✅ completed  | `scripts/download_bagel.py`, `scripts/sanity_inference.py`          | runtime DL of BAGEL-7B            |
| P2    | ✅ completed  | `src/bagel_sbsr/sbsr.py`, attention hook, unit tests                | SBSR core + saliency rollout      |
| P3    | ✅ committed  | `scripts/train_s1.py`, `scripts/coyo_dataloader.py`, `configs/s1.yaml` | S1 LoRA warmup; dry-run green; full train requires GPU |
| P4    | ✅ committed  | `scripts/train_s2.py`, `configs/s2.yaml`                            | iMF + DMD2 dual; both dry-runs green; FID/CLIP gate |
| P5    | ✅ committed  | `scripts/train_s3.py`, `configs/s3.yaml`                            | integration FT; dry-run green     |
| P6    | ✅ committed  | `scripts/eval.py`, `configs/eval.yaml`, `docs/EVAL.md`              | FID / CLIP-T / GenEval / T2I-CompBench / HPSv2.1 |
| P7    | ✅ committed  | `comfyui/nodes.py`, `src/bagel_sbsr/pipeline_bagel_sbsr.py`, `demo/app.py` | distribution layer     |
| P8    | ✅ committed  | `RELEASE_NOTES.md`, `MODEL_CARD.md`, `papers/preprint.tex`, `MANUAL.md` | v0.1.0.dev ready; weights need user-side training |

## Code-level vs. release-level completion

- **Code-level** (this repo): P0 → P8 all committed and unit-tested.
- **Release-level** (weights on HF, v0.1.0 tag): requires the user-side
  training run described in [MANUAL.md](../MANUAL.md). RunPod compute
  (~$2,880 / ~10 days) cannot be incurred from Claude — by design (R11).

## Safety net (per arch spec §安全装置)

- NaN-detect: rollback after 100 consecutive NaN/Inf steps (impl in `train_s1.py` `_real_train`)
- Checkpoint sha256 verification on save/resume (impl: `_sha256_file`, `.safetensors.sha256` sidecars)
- RunPod preemption auto-relaunch (impl: `launch_runpod.py` poll loop)
- COYO URL deadlist monitoring (>30% 404 → mirror fallback) (impl: `coyo_dataloader.stream_coyo`)
- 3 consecutive same-phase same-error → user notification, otherwise continue
- Failure museum: `experiments/_wip/<phase>_<timestamp>/` per R8 convention
- safetensors-only checkpoints (no pickle) for supply-chain safety
