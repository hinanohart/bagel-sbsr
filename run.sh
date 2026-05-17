#!/usr/bin/env bash
# BAGEL-SBSR — end-to-end runner (1 script, all-in-one)
# Usage: bash run.sh
#
# Prerequisites the user must provide *before* running this script:
#   HF_TOKEN          — https://huggingface.co/settings/tokens (read+write)
#   RUNPOD_API_KEY    — https://runpod.io/console/user/settings
#   WANDB_API_KEY     — https://wandb.ai/authorize  (optional but recommended)
#
# Provide them via env, NOT inline:
#   export HF_TOKEN='hf_xxx...'
#   export RUNPOD_API_KEY='rpa_xxx...'
#   export WANDB_API_KEY='xxx...'   # optional
#   bash run.sh
#
# What this script does (in order):
#   1. Env-var preflight (fail fast)
#   2. uv sync (Python deps incl. dev)
#   3. Vendor BAGEL upstream + download weights (~15 GB)
#   4. CPU smoke tests + 5 dry-runs
#   5. Launch full train pipeline on RunPod 4×H100 (S1 → S2 dual → gate → S3 → eval)
#   6. Upload trained weights to your HF account
#   7. (Optional) draft a GitHub release
#
# Honest caveat: step 5 will currently HALT at the first call to
# `_bagel_forward_adapter` (NotImplementedError) — see "Truly manual" below.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

log() { printf '\033[1;36m[run.sh]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[run.sh ERROR]\033[0m %s\n' "$*" >&2; }

# ---- 1. preflight -----------------------------------------------------------
log "Step 1/7  preflight: env vars"
: "${HF_TOKEN:?HF_TOKEN is required — see https://huggingface.co/settings/tokens}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY is required — see https://runpod.io/console/user/settings}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
HF_USER="${HF_USER:-}"   # for final weight upload; auto-resolved if empty

command -v uv >/dev/null || { err "uv not installed — https://docs.astral.sh/uv/getting-started/installation"; exit 1; }
command -v git >/dev/null || { err "git not installed"; exit 1; }

# ---- 2. deps ----------------------------------------------------------------
log "Step 2/7  uv sync (deps)"
uv sync --extra dev --extra train --prerelease=allow

# ---- 3. vendor + weights ----------------------------------------------------
log "Step 3/7  vendor BAGEL upstream + download weights"
if [ ! -d vendor/bagel-upstream ]; then
  bash scripts/install_bagel_src.sh
fi
if [ ! -d weights/bagel-7b-mot ]; then
  uv run scripts/download_bagel.py --dest weights/bagel-7b-mot
fi

# ---- 4. smoke + dry-run -----------------------------------------------------
log "Step 4/7  smoke tests + dry-runs"
uv run pytest -q -m smoke
uv run scripts/train_s1.py --config configs/s1.yaml --dry-run
uv run scripts/train_s2.py --config configs/s2.yaml --track imf  --dry-run
uv run scripts/train_s2.py --config configs/s2.yaml --track dmd2 --dry-run
uv run scripts/train_s3.py --config configs/s3.yaml --dry-run
uv run scripts/eval.py    --config configs/eval.yaml --dry-run

# ---- 5. full training -------------------------------------------------------
log "Step 5/7  full pipeline on RunPod 4×H100 (S1→S2 dual→gate→S3→eval)"
log "          this is the step that will HALT until v0.1.1 adapter lands"
log "          (see 'Truly manual' below). Re-running this script after the"
log "          adapter is implemented will resume from the same step."
bash scripts/launch_full.sh

# ---- 6. weight upload -------------------------------------------------------
log "Step 6/7  upload trained weights to HF Hub"
if [ -z "$HF_USER" ]; then
  HF_USER=$(uv run python -c "from huggingface_hub import whoami; import os; print(whoami(token=os.environ['HF_TOKEN'])['name'])")
fi
uv run python -c "
from huggingface_hub import HfApi, create_repo
import os
api = HfApi(token=os.environ['HF_TOKEN'])
repo = f'${HF_USER}/bagel-sbsr-v0.1.0'
create_repo(repo, exist_ok=True, repo_type='model', private=False)
api.upload_folder(folder_path='runs/s3/final', repo_id=repo, repo_type='model')
print(f'uploaded to https://huggingface.co/{repo}')
"

# ---- 7. github release ------------------------------------------------------
log "Step 7/7  draft GitHub release"
if command -v gh >/dev/null; then
  gh release create v0.1.0 \
    --repo hinanohart/bagel-sbsr \
    --title 'BAGEL-SBSR v0.1.0' \
    --notes-file RELEASE_NOTES.md \
    --draft || log "release already exists or auth missing — skip"
fi

log "DONE.  Repo: https://github.com/hinanohart/bagel-sbsr"
log "       Model: https://huggingface.co/${HF_USER}/bagel-sbsr-v0.1.0"
