#!/usr/bin/env bash
# launch_full.sh — orchestrate P3 (S1) → P4 (S2 dual) → P5 (S3) → P6 (eval)
#
# Reads HF_TOKEN, RUNPOD_API_KEY, WANDB_API_KEY (optional) from env.
# Refuses to launch if any required secret is missing or if the cumulative
# RunPod cost would exceed `safety.budget_hard_ceiling_usd` in any config.

set -euo pipefail

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "FATAL: HF_TOKEN not set. See MANUAL.md step 2." >&2
    exit 1
fi
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    echo "FATAL: RUNPOD_API_KEY not set. See MANUAL.md step 2." >&2
    exit 1
fi

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

RUNS_DIR="runs"
mkdir -p "$RUNS_DIR"

echo "[$(date -u +%FT%TZ)] Stage 1 — SBSR LoRA warm-up"
uv run scripts/launch_runpod.py \
    --config configs/s1.yaml \
    --script scripts/train_s1.py

S1_CKPT=$(ls -1 -t runs/s1/ckpt-*.safetensors 2>/dev/null | head -1 || true)
if [[ -z "$S1_CKPT" ]]; then
    echo "FATAL: no S1 checkpoint produced." >&2
    exit 1
fi
echo "  -> S1 winner: $S1_CKPT"

echo "[$(date -u +%FT%TZ)] Stage 2 — iMF + DMD2 dual distillation"
uv run scripts/launch_runpod.py \
    --config configs/s2.yaml \
    --script scripts/train_s2.py
# In production, the two tracks are launched as independent pods. The
# FID/CLIP gate inside train_s2.py decides which track continues to 80k
# steps; the loser is aborted by the gate exit code.

S2_BEST=""
for track in imf dmd2; do
    ckpt=$(ls -1 -t runs/s2/$track/ckpt-*.safetensors 2>/dev/null | head -1 || true)
    score_file=$(ls -1 -t runs/s2/$track/gate-*.json 2>/dev/null | head -1 || true)
    if [[ -n "$ckpt" && -n "$score_file" ]]; then
        score=$(python -c "import json,sys;print(json.load(open('$score_file'))['fid_clip_score'])")
        echo "  $track score=$score ($ckpt)"
        if [[ -z "$S2_BEST" ]]; then
            S2_BEST="$ckpt"; BEST_SCORE="$score"
        else
            python -c "import sys;sys.exit(0 if float('$score') < float('$BEST_SCORE') else 1)" \
                && { S2_BEST="$ckpt"; BEST_SCORE="$score"; }
        fi
    fi
done
if [[ -z "$S2_BEST" ]]; then
    echo "FATAL: neither iMF nor DMD2 produced a checkpoint." >&2
    exit 1
fi
echo "  -> S2 winner: $S2_BEST (score=$BEST_SCORE)"

echo "[$(date -u +%FT%TZ)] Stage 3 — integrated fine-tune"
uv run scripts/launch_runpod.py \
    --config configs/s3.yaml \
    --script scripts/train_s3.py

S3_CKPT=$(ls -1 -t runs/s3/ckpt-*.safetensors 2>/dev/null | head -1 || true)
if [[ -z "$S3_CKPT" ]]; then
    echo "FATAL: no S3 checkpoint produced." >&2
    exit 1
fi
echo "  -> S3 final: $S3_CKPT"

echo "[$(date -u +%FT%TZ)] Stage 6 — evaluation"
uv run scripts/eval.py \
    --config configs/eval.yaml \
    --ckpt "$S3_CKPT"

echo "[$(date -u +%FT%TZ)] launch_full.sh complete."
echo "  S1: $S1_CKPT"
echo "  S2: $S2_BEST  (score $BEST_SCORE)"
echo "  S3: $S3_CKPT"
echo "  eval: runs/eval/*/scores.json"
echo ""
echo "Next: see MANUAL.md step 5 to publish."
