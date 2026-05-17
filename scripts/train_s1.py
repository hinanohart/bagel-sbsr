"""Stage 1 — SBSR warm-up (LoRA on BAGEL gen-expert q/v + SBSR scalars).

This script is *complete enough to dry-run* (build optimizer, take one step
of dummy data) but the heavy data path (COYO-700M streaming + BAGEL packed
collator) is wired through a `--dry-run` flag so it can be exercised on a
laptop CPU without weights. Real training requires:

    HF_TOKEN=...            (for snapshot_download of BAGEL-7B-MoT)
    PYTHONPATH=vendor/bagel-upstream

Usage:
    uv run scripts/train_s1.py --config configs/s1.yaml [--dry-run]

Exit codes:
    0  success
   77  environment incomplete (weights/vendor/CUDA absent) — CI skip
    1  actual failure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="single forward+backward on a synthetic batch, then exit",
    )
    p.add_argument("--resume", type=Path, default=None, help="optional checkpoint to resume from")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _check_env(cfg: dict, dry_run: bool) -> int | None:
    """Return None to continue, or an exit code (77 skip, 1 error)."""
    weights = Path(cfg["model"]["weights_dir"])
    vendor = Path(cfg["model"]["vendor_dir"])
    if not dry_run:
        if not weights.exists() or not (weights / "config.json").exists():
            print(f"SKIP: weights missing at {weights}", file=sys.stderr)
            return 77
        if not vendor.exists() or not (vendor / ".git").exists():
            print(f"SKIP: vendor missing at {vendor}", file=sys.stderr)
            return 77
    try:
        import torch
    except ImportError:
        print("SKIP: torch missing", file=sys.stderr)
        return 77
    if not dry_run and not torch.cuda.is_available():
        print("SKIP: no CUDA device", file=sys.stderr)
        return 77
    return None


def _build_model_and_sbsr(cfg: dict):
    """Build BAGEL, freeze it, install LoRA + SBSR on gen-expert q/v."""
    from peft import LoraConfig, get_peft_model

    from bagel_sbsr import patch_bagel
    from scripts.saliency_provider import SigLIP2RolloutProvider  # type: ignore[import-not-found]

    # Reuse the upstream BAGEL setup from sanity_inference._build_inferencer.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sanity_inference import _build_inferencer  # type: ignore[import-not-found]

    inferencer = _build_inferencer(Path(cfg["model"]["weights_dir"]))
    model = inferencer.model

    for p in model.parameters():
        p.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, lora_cfg)

    provider = SigLIP2RolloutProvider(model)
    patch_bagel(
        model,
        lambda_init=cfg["sbsr"]["lambda_init"],
        mu_init=cfg["sbsr"]["mu_init"],
        top_k=cfg["sbsr"]["top_k"],
        saliency_provider=provider,
        learnable=cfg["sbsr"]["learnable"],
    )
    return model, inferencer


def _dry_run_step(cfg: dict) -> int:
    """One synthetic forward+backward to verify wire-up without weights."""
    import torch
    import torch.nn as nn

    from bagel_sbsr import SBSR

    print("DRY-RUN: synthetic SBSR step (no BAGEL weights needed)")
    sbsr = SBSR(
        lambda_init=cfg["sbsr"]["lambda_init"],
        mu_init=cfg["sbsr"]["mu_init"],
        top_k=cfg["sbsr"]["top_k"],
        learnable=cfg["sbsr"]["learnable"],
    )
    opt = torch.optim.AdamW(sbsr.parameters(), lr=cfg["train"]["learning_rate"])

    B, T = 2, 16
    saliency = torch.rand(B, T)
    logits = torch.randn(B, T, T, requires_grad=True)
    out = sbsr(logits, saliency, apply_top_k=False)
    target = torch.randn_like(out)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()
    opt.step()
    print(
        f"DRY-RUN OK: loss={loss.item():.4f}, lambda={float(sbsr.lambda_):.4f}, mu={float(sbsr.mu_):.4f}"
    )
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    rc = _check_env(cfg, args.dry_run)
    if rc is not None:
        return rc

    if args.dry_run:
        return _dry_run_step(cfg)

    # Full training path: implemented by P3 follow-up commits.
    print(
        "ERROR: full S1 training entrypoint is staged. Run with --dry-run for the "
        "wire-up smoke; full COYO-700M streaming, Accelerate launcher, NaN-detect "
        "rollback, and checkpoint sharding are slated for the next P3 commit.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
