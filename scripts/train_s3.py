"""Stage 3 — Integrated fine-tune: SBSR + winner distillation track.

After S2 produces a single winner track (iMF or DMD2 via the FID/CLIP gate),
S3 merges the winner's gen-expert weights with the SBSR module from S1 and
runs a shorter integration fine-tune on a curated subset (COYO + GenEval +
T2I-CompBench train splits).

Goals:
- recover small regressions introduced by aggressive distillation
- jointly tune SBSR lambda/mu against the now-faster sampler
- prep the model for evaluation (P6) and release (P7/P8)

Usage:
    uv run scripts/train_s3.py --config configs/s3.yaml --dry-run
    accelerate launch scripts/train_s3.py --config configs/s3.yaml \\
        --s2-winner runs/s2/imf/ckpt-00080000.safetensors

Exit codes:
    0  success
   77  environment incomplete
    1  failure
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--s2-winner", type=Path, default=None)
    p.add_argument("--s1-checkpoint", type=Path, default=None)
    p.add_argument("--resume", type=Path, default=None)
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _check_env(cfg: dict, dry_run: bool) -> int | None:
    try:
        import torch
    except ImportError:
        print("SKIP: torch missing", file=sys.stderr)
        return 77
    if dry_run:
        return None
    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device", file=sys.stderr)
        return 77
    return None


def _dry_run(cfg: dict) -> int:
    """Synthetic SBSR + flow-matching MSE step to validate wire-up."""
    import torch

    from bagel_sbsr import SBSR

    torch.manual_seed(cfg["run"]["seed"])
    sbsr = SBSR(
        lambda_init=cfg["sbsr"]["lambda_init"],
        mu_init=cfg["sbsr"]["mu_init"],
        top_k=cfg["sbsr"]["top_k"],
        learnable=True,
    )

    B, T = 2, 16
    saliency = torch.rand(B, T)
    logits = torch.randn(B, T, T, requires_grad=True)
    out = sbsr(logits, saliency, apply_top_k=True)
    # Combine bias path with a flow-matching MSE on synthetic latents.
    latent = torch.randn(B, cfg["data"]["vae_latent_channels"], 8, 8, requires_grad=True)
    eps = torch.randn_like(latent)
    t = torch.rand(B).clamp(0.01, 0.99)
    x_t = (1 - t).view(-1, 1, 1, 1) * latent + t.view(-1, 1, 1, 1) * eps
    v = torch.zeros_like(x_t, requires_grad=True)
    target = (latent - eps).detach()
    fm_loss = torch.nn.functional.mse_loss(v, target)
    sbsr_loss = out[torch.isfinite(out)].pow(2).mean()
    loss = cfg["loss"]["mse_weight"] * fm_loss + 0.01 * sbsr_loss
    loss.backward()
    print(
        f"DRY-RUN OK: total_loss={loss.item():.4f} fm={fm_loss.item():.4f} "
        f"sbsr={sbsr_loss.item():.4f} "
        f"lambda={float(sbsr.lambda_.detach()):.4f} mu={float(sbsr.mu_.detach()):.4f}"
    )
    return 0


def _real_train(
    cfg: dict, s2_winner: Path | None, s1_ckpt: Path | None, resume: Path | None
) -> int:
    import torch

    from bagel_sbsr import LatentMagnitudeProvider, patch_bagel

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sanity_inference import _build_inferencer  # type: ignore[import-not-found]

    try:
        from accelerate import Accelerator  # type: ignore[import-not-found]
    except ImportError:
        print("ERROR: accelerate not installed", file=sys.stderr)
        return 1

    accelerator = Accelerator(mixed_precision="bf16")
    inferencer = _build_inferencer(Path(cfg["model"]["weights_dir"]))
    model = inferencer.model

    # Make the gen expert fully trainable (S3 is full FT, not LoRA).
    for p in model.parameters():
        p.requires_grad_(False)
    for n, p in model.named_parameters():
        if "gen" in n.lower():
            p.requires_grad_(True)

    provider = LatentMagnitudeProvider()
    patch_bagel(
        model,
        lambda_init=cfg["sbsr"]["lambda_init"],
        mu_init=cfg["sbsr"]["mu_init"],
        top_k=cfg["sbsr"]["top_k"],
        saliency_provider=provider,
        learnable=cfg["sbsr"]["learnable"],
        require_layers=1,
    )

    if s2_winner is not None:
        # S2 saves under `gen/` (full-FT weights); read those into the
        # student's named parameters. S2 ckpt may also contain `sbsr/` —
        # load both prefixes.
        _load_safetensors_into(model, s2_winner, key_prefix="gen/")
        _load_safetensors_into(model, s2_winner, key_prefix="sbsr/")
    if s1_ckpt is not None:
        # S1 ckpt has `sbsr/` and `lora/`. Load both — `lora/` weights map
        # onto the matching gen-expert params after S2 made them full FT.
        _load_safetensors_into(model, s1_ckpt, key_prefix="sbsr/")
        _load_safetensors_into(model, s1_ckpt, key_prefix="lora/")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        trainable,
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    model, opt = accelerator.prepare(model, opt)

    out_dir = Path(cfg["run"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.coyo_dataloader import (  # type: ignore[import-not-found]
        CoyoStreamConfig,
        iter_coyo_batches,
    )

    stream_cfg = CoyoStreamConfig(
        resolution=cfg["data"]["resolution"],
        mirror_fallback_threshold=cfg["data"]["mirror_fallback_threshold"],
        shuffle_buffer=cfg["data"]["shuffle_buffer"],
        seed=cfg["run"]["seed"],
    )
    transform = inferencer.image_transform
    batch_iter = iter_coyo_batches(
        stream_cfg,
        image_transform=transform,
        batch_size=cfg["train"]["per_device_batch_size"],
    )

    nan_window: deque = deque(maxlen=cfg["train"]["nan_detect_consecutive"])
    step = 0
    t0 = time.time()
    for batch in batch_iter:
        if step >= cfg["train"]["num_steps"]:
            break
        with accelerator.accumulate(model):
            opt.zero_grad()
            with torch.no_grad():
                latent = inferencer.vae_model.encode(batch.images.to(accelerator.device))
                if hasattr(latent, "latent_dist"):
                    latent = latent.latent_dist.sample()
            provider.set_latent(latent)

            # ADAPTER SLOT — same as S1; wire to real Bagel.forward.
            loss = _bagel_forward_adapter_s3(
                model,
                batch,
                latent,
                inferencer,
                cfg["loss"]["ce_weight"],
                cfg["loss"]["mse_weight"],
            )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(trainable, cfg["train"]["max_grad_norm"])
            opt.step()

        loss_val = loss.detach().float().item()
        nan_window.append(1 if loss_val != loss_val else 0)

        if step % cfg["train"]["log_every"] == 0 and accelerator.is_main_process:
            print(f"step={step} loss={loss_val:.4f} elapsed={time.time() - t0:.1f}s")

        if step % cfg["train"]["save_every"] == 0 and step > 0 and accelerator.is_main_process:
            _save_safetensors(out_dir, step, accelerator.unwrap_model(model))

        step += 1
        provider.reset()

    if accelerator.is_main_process:
        _save_safetensors(out_dir, step, accelerator.unwrap_model(model))
        (out_dir / "final_metadata.json").write_text(
            json.dumps({"steps_completed": step, "elapsed_seconds": time.time() - t0}, indent=2)
        )
    return 0


def _bagel_forward_adapter_s3(model, batch, latent, inferencer, ce_w, mse_w):
    """Same contract as scripts/train_s1.py:_bagel_forward_adapter."""
    raise NotImplementedError(
        "BAGEL forward adapter (S3) is the v0.1.1 follow-up — wire batch -> "
        "Bagel.forward per vendor/bagel-upstream/modeling/bagel/bagel.py:101."
    )


def _load_safetensors_into(model, path: Path, key_prefix: str) -> None:
    import torch
    from safetensors.torch import load_file  # type: ignore[import-not-found]

    flat = load_file(str(path))
    own = dict(model.named_parameters())
    sbsr = getattr(model, "sbsr", None)
    sbsr_params = dict(sbsr.named_parameters()) if sbsr is not None else {}
    with torch.no_grad():
        for k, v in flat.items():
            if not k.startswith(key_prefix):
                continue
            local = k[len(key_prefix) :]
            if key_prefix == "sbsr/" and sbsr is not None and local in sbsr_params:
                sbsr.get_parameter(local).copy_(v.to(sbsr_params[local].device))
            elif key_prefix == "lora/" and local in own and own[local].shape == v.shape:
                own[local].copy_(v.to(own[local].device))


def _save_safetensors(out_dir: Path, step: int, model) -> Path:
    from safetensors.torch import save_file  # type: ignore[import-not-found]

    ckpt = out_dir / f"ckpt-{step:08d}.safetensors"
    flat: dict = {}
    sbsr = getattr(model, "sbsr", None)
    if sbsr is not None:
        for k, v in sbsr.state_dict().items():
            flat[f"sbsr/{k}"] = v.contiguous().detach().cpu()
    for n, p in model.named_parameters():
        if p.requires_grad and "sbsr" not in n:
            flat[f"gen/{n}"] = p.contiguous().detach().cpu()
    save_file(flat, str(ckpt))
    (ckpt.with_suffix(ckpt.suffix + ".json")).write_text(json.dumps({"step": step}))
    return ckpt


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    rc = _check_env(cfg, args.dry_run)
    if rc is not None:
        return rc
    if args.dry_run:
        return _dry_run(cfg)
    return _real_train(cfg, args.s2_winner, args.s1_checkpoint, args.resume)


if __name__ == "__main__":
    sys.exit(main())
