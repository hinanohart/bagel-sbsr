# Rolling phase tracker

This file tracks the autonomous build pipeline (P0 → P8). Updated at the end of each phase.

| Phase | Status        | Artifact                                                            | Notes                             |
| ----- | ------------- | ------------------------------------------------------------------- | --------------------------------- |
| P0    | ✅ completed  | repo init / LICENSE / pyproject / CI / docs stubs                   | scaffold only, no impl yet        |
| P1    | 🟡 in flight  | `scripts/download_bagel.py`, `scripts/sanity_inference.py`          | runtime DL of BAGEL-7B            |
| P2    | ⏳ pending    | `src/bagel_sbsr/sbsr.py`, attention hook, unit tests                | SBSR core (~30 lines + saliency)  |
| P3    | ⏳ pending    | `scripts/train_s1.py`                                               | LoRA r=16 warmup                  |
| P4    | ⏳ pending    | `scripts/train_s2.py`                                               | iMF + DMD2 dual distillation      |
| P5    | ⏳ pending    | `scripts/train_s3.py`                                               | integration FT                    |
| P6    | ⏳ pending    | `scripts/eval.py`, `src/bagel_sbsr/eval/`                           | FID/CLIP-T/GenEval/T2I-CompBench  |
| P7    | ⏳ pending    | `comfyui/nodes.py`, `src/bagel_sbsr/pipeline_bagel_sbsr.py`, `demo/`| distribution                      |
| P8    | ⏳ pending    | `RELEASE_NOTES.md`, `papers/preprint.tex`, `MODEL_CARD.md`          | v0.1.0 release                    |

## Safety net (per arch spec §安全装置)

- NaN-detect: rollback after 100 consecutive NaN/Inf steps
- Checkpoint sha256 verification on save/resume
- RunPod preemption auto-relaunch
- COYO URL deadlist monitoring (>30% 404 → mirror fallback)
- 3 consecutive same-phase same-error → user notification, otherwise continue
- Failure museum: `experiments/_wip/<phase>_<timestamp>/` per R8 convention
