# MANUAL.md — User-required manual steps

This is the **single source of truth** for everything the user must do
by hand. All code-side automation (P0 → P8) is committed and tested;
the steps below are the irreducible manual prerequisites because they
involve secrets, third-party account creation, or external GPU billing
that the toolchain deliberately cannot do on your behalf.

> If you only run the dry-runs (no GPU, no secrets), you can stop after
> step 1. The remaining steps are only needed to actually train and
> publish weights.

---

## 1. Local clone + smoke (no secrets, no GPU)

```bash
git clone https://github.com/hinanohart/bagel-sbsr.git
cd bagel-sbsr
uv sync                                  # installs Python deps
uv run pytest -q -m smoke                # 30 passed expected
uv run scripts/train_s1.py --config configs/s1.yaml --dry-run
uv run scripts/train_s2.py --config configs/s2.yaml --track imf --dry-run
uv run scripts/train_s2.py --config configs/s2.yaml --track dmd2 --dry-run
uv run scripts/train_s3.py --config configs/s3.yaml --dry-run
uv run scripts/eval.py --config configs/eval.yaml --dry-run
```

All five dry-runs print `DRY-RUN OK` / wrote scores.json. If any fails,
file an issue.

---

## 2. Set required secrets (env vars only — Claude never sees them)

```bash
# Replace `<your-token-here>` with the actual value from each provider's settings page.
export HF_TOKEN='<your-huggingface-token-from-settings-tokens>'
export RUNPOD_API_KEY='<your-runpod-api-key-from-user-settings>'
export WANDB_API_KEY='<your-wandb-key-optional>'        # only for run logging
```

Token sources:
- HuggingFace: https://huggingface.co/settings/tokens
- RunPod: https://www.runpod.io/console/user/settings
- Weights & Biases: https://wandb.ai/settings (optional)

Recommendation: write these to `~/.bagel-sbsr.env` (chmod 600) and
`source ~/.bagel-sbsr.env` once per shell session.

---

## 3. Pull the BAGEL upstream code + weights (locally, one-time)

```bash
scripts/install_bagel_src.sh             # clones BAGEL pinned to a known rev
uv run scripts/download_bagel.py --dest weights/bagel-7b-mot
uv run scripts/sanity_inference.py \
    --weights weights/bagel-7b-mot \
    --prompt "a calico cat on a windowsill" \
    --out-dir sanity_out
```

`sanity_out/t2i.png` and `sanity_out/caption.txt` should be produced.

---

## 4. Launch the full training pipeline (RunPod)

Single command launches S1 → S2 (dual) → gate → winner → S3 → eval:

```bash
bash scripts/launch_full.sh
```

Internally this calls `scripts/launch_runpod.py` per stage, waits on
the FID/CLIP gate to decide iMF vs DMD2, and emits stage-by-stage
checkpoints to `runs/`. Each stage's `launch_runpod.py` enforces the
per-stage `max_cost_usd` cap (S1 \$800, S2 \$1,700, S3 \$500). The
wrapper aborts the chain if any stage exits non-zero, so a cap breach
in stage *N* stops stages *N+1..*.

**Adapter caveat (v0.1.0.dev):** the BAGEL packed-sequence collator
that bridges `(image, caption)` batches to `Bagel.forward` (see
`docs/TRAINING.md` §"BAGEL forward adapter") is intentionally not
written without GPU-side iteration. `train_s1/s2/s3.py` raise
`NotImplementedError` from the adapter slot on the real-train path
until v0.1.1. Until then, this command will exit 1 at the first
adapter call, which is the honest behavior.

Total wall-clock: ~10 days on 4×H100. Total spend depends on the GPU
pool you draw from at the time you run:

- H100 SXM **on-demand** (~$2.69-2.99/h × 4 GPUs × 240 h): ~$2,580-$2,870
- H100 SXM **spot** when available: roughly half of on-demand
- H100 **PCIe** on-demand (~$1.99/h × 4 × 240 h): ~$1,910

`safety.budget_hard_ceiling_usd` (default $4,000) is enforced by
`launch_runpod.py` — the script polls accumulated cost and calls
`runpod.terminate_pod` if the ceiling is exceeded. **The runtime is in
your account — Claude cannot launch this on your behalf, by design
(R11 / R13).**

---

## 5. Release the trained weights (after step 4 completes)

```bash
# Sanity-check the integrated checkpoint
uv run scripts/eval.py \
    --config configs/eval.yaml \
    --ckpt runs/s3/ckpt-00020000.safetensors

# Push to HF
huggingface-cli upload hinanohart/bagel-sbsr-v0.1.0 runs/s3/

# Cut the GitHub release
gh release create v0.1.0 \
    --title "BAGEL-SBSR v0.1.0" \
    --notes-file RELEASE_NOTES.md \
    runs/s3/ckpt-00020000.safetensors{,.sha256,.json}
```

---

## What is *intentionally* manual

| Step | Why it must stay manual |
|---|---|
| Setting HF_TOKEN / RUNPOD_API_KEY | R11: Claude must never read or echo API tokens. Env vars only. |
| Running `scripts/launch_full.sh` | RunPod billing goes through *your* account, not Claude's. R13. |
| Approving the v0.1.0 GitHub release | OSS publication is irreversible (cache, mirrors, downstream forks). User signs off. |

Everything else (code, configs, tests, dry-runs, secret-handling
plumbing, license audit, model card, preprint outline) is shipped.
