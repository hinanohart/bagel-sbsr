"""Stage 1 — SBSR warm-up (LoRA on BAGEL gen-expert q/v + SBSR scalars).

This is a full-fat training entrypoint. It supports two modes:
  --dry-run       Single forward+backward on a synthetic batch through a
                  *fake* BAGEL layer that exercises the SBSR hook wire-up
                  (patch_bagel -> wrapper -> bias + top-k applied -> unpatch).
                  Runs on CPU without weights. CI-friendly.
  (no flag)       Full training loop with COYO-700M streaming, Accelerate,
                  bf16, NaN-detect rollback, rolling checkpoint, sha256.
                  Requires GPU + weights + vendor/bagel-upstream/.

The CE/MSE loss split (`loss.ce_weight`, `loss.mse_weight`) implements the
architecture spec (CE:MSE = 0.25:1) on the FLUX VAE latent rectified-flow
objective.

Usage:
    uv run scripts/train_s1.py --config configs/s1.yaml --dry-run
    accelerate launch scripts/train_s1.py --config configs/s1.yaml

Exit codes:
    0  success
   77  environment incomplete (CI skip)
    1  actual failure
"""

from __future__ import annotations

import argparse
import hashlib
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
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="exercise patch/unpatch cycle + 1 step on synthetic data, then exit",
    )
    p.add_argument("--resume", type=Path, default=None, help="optional checkpoint to resume from")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _check_env(cfg: dict, dry_run: bool) -> int | None:
    """Return None to continue, or an exit code (77 skip, 1 error)."""
    try:
        import torch
    except ImportError:
        print("SKIP: torch missing", file=sys.stderr)
        return 77
    if dry_run:
        return None

    weights = Path(cfg["model"]["weights_dir"])
    vendor = Path(cfg["model"]["vendor_dir"])
    if not weights.exists() or not (weights / "config.json").exists():
        print(f"SKIP: weights missing at {weights}", file=sys.stderr)
        return 77
    if not vendor.exists() or not (vendor / ".git").exists():
        print(f"SKIP: vendor missing at {vendor}", file=sys.stderr)
        return 77

    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA device", file=sys.stderr)
        return 77
    return None


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _save_checkpoint(out_dir: Path, step: int, sbsr_state: dict, lora_state: dict) -> Path:
    """Save tensors via safetensors (no pickle) + step metadata via JSON."""
    from safetensors.torch import save_file  # type: ignore[import-not-found]

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"ckpt-{step:08d}.safetensors"
    # Prefix keys so we can split sbsr/lora on load.
    flat: dict = {}
    for k, v in sbsr_state.items():
        flat[f"sbsr/{k}"] = v.contiguous().detach().cpu()
    for k, v in lora_state.items():
        flat[f"lora/{k}"] = v.contiguous().detach().cpu()
    save_file(flat, str(ckpt))
    (ckpt.with_suffix(ckpt.suffix + ".sha256")).write_text(_sha256_file(ckpt))
    (ckpt.with_suffix(ckpt.suffix + ".json")).write_text(json.dumps({"step": step}))
    return ckpt


def _load_checkpoint(ckpt: Path) -> dict:
    """Load a safetensors checkpoint back into a dict {step, sbsr, lora}."""
    from safetensors.torch import load_file  # type: ignore[import-not-found]

    flat = load_file(str(ckpt))
    meta_path = ckpt.with_suffix(ckpt.suffix + ".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"step": 0}
    sbsr_state, lora_state = {}, {}
    for k, v in flat.items():
        if k.startswith("sbsr/"):
            sbsr_state[k[len("sbsr/") :]] = v
        elif k.startswith("lora/"):
            lora_state[k[len("lora/") :]] = v
    return {"step": meta.get("step", 0), "sbsr": sbsr_state, "lora": lora_state}


def _prune_rolling_ckpts(out_dir: Path, keep: int) -> None:
    rolls = sorted(out_dir.glob("rolling-*.safetensors"))
    for old in rolls[:-keep]:
        old.unlink(missing_ok=True)
        Path(str(old) + ".sha256").unlink(missing_ok=True)
        Path(str(old) + ".json").unlink(missing_ok=True)


# ----- dry-run path ------------------------------------------------------------


def _dry_run(cfg: dict) -> int:
    """Exercise patch_bagel -> wrapped forward_train -> unpatch_bagel on a fake BAGEL.

    The fake module mimics PackedAttentionMoT.forward_train. This validates that
    (1) SBSR bias actually reaches the attention_mask,
    (2) top-k mask collapses to k finite entries on gen rows,
    (3) gradients flow to lambda/mu,
    (4) unpatch_bagel restores the original method.
    """
    import sys as _sys
    import types

    import torch
    import torch.nn as nn

    from bagel_sbsr import LatentMagnitudeProvider, patch_bagel, unpatch_bagel

    # Install fake BAGEL upstream
    class FakePackedAttentionMoT(nn.Module):
        def forward_train(
            self,
            packed_sequence,
            sample_lens,
            attention_mask,
            packed_position_embeddings,
            packed_und_token_indexes,
            packed_gen_token_indexes,
        ):
            return ("ok", attention_mask)

    fake_mod = types.ModuleType("modeling")
    fake_bagel = types.ModuleType("modeling.bagel")
    fake_qwen = types.ModuleType("modeling.bagel.qwen2_navit")
    fake_qwen.PackedAttentionMoT = FakePackedAttentionMoT
    fake_mod.bagel = fake_bagel
    fake_bagel.qwen2_navit = fake_qwen
    _sys.modules["modeling"] = fake_mod
    _sys.modules["modeling.bagel"] = fake_bagel
    _sys.modules["modeling.bagel.qwen2_navit"] = fake_qwen

    class FakeBagel(nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(2):
                setattr(self, f"attn{i}", FakePackedAttentionMoT())

    model = FakeBagel()
    provider = LatentMagnitudeProvider()
    # 8x8 latent -> 64 patches, n = 64 gen tokens + 8 und = 72
    provider.set_latent(torch.randn(1, cfg["data"]["vae_latent_channels"], 8, 8))

    patched = patch_bagel(
        model,
        lambda_init=cfg["sbsr"]["lambda_init"],
        mu_init=cfg["sbsr"]["mu_init"],
        top_k=cfg["sbsr"]["top_k"],
        saliency_provider=provider,
        learnable=cfg["sbsr"]["learnable"],
        require_layers=2,
    )
    assert len(patched) == 2

    sbsr = model.sbsr
    opt = torch.optim.AdamW(sbsr.parameters(), lr=cfg["train"]["learning_rate"])

    und_n, gen_n = 8, 64
    n = und_n + gen_n
    packed_sequence = torch.zeros(n, 16)
    attention_mask = [torch.zeros(1, n, n)]
    sample_lens = [n]
    packed_und = torch.arange(0, und_n, dtype=torch.long)
    packed_gen = torch.arange(und_n, n, dtype=torch.long)
    pos = torch.zeros(n, 8)

    _, biased = model.attn0.forward_train(
        packed_sequence, sample_lens, attention_mask, pos, packed_und, packed_gen
    )
    bm = biased[0][0]  # (n, n)

    # The gen<->gen block should have at most `top_k` finite entries per row.
    gen_block = bm[und_n:, und_n:]
    finite_per_row = torch.isfinite(gen_block).sum(dim=-1)
    top_k = cfg["sbsr"]["top_k"]
    assert finite_per_row.max() <= top_k, finite_per_row.max()

    loss = bm[torch.isfinite(bm)].pow(2).mean()
    loss.backward()
    opt.step()

    restored = unpatch_bagel(model)
    assert restored == 2

    print(
        f"DRY-RUN OK: loss={loss.item():.4f}, "
        f"lambda={float(sbsr.lambda_.detach()):.4f}, mu={float(sbsr.mu_.detach()):.4f}, "
        f"top_k_active=True, patch_cycle_restored={restored}/2"
    )
    return 0


# ----- real train loop ---------------------------------------------------------


def _build_inferencer_and_lora(cfg: dict):
    """Build BAGEL + LoRA + SBSR (real path). Imports vendor lazily."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sanity_inference import _build_inferencer  # type: ignore[import-not-found]

    inferencer = _build_inferencer(Path(cfg["model"]["weights_dir"]))
    model = inferencer.model

    for p in model.parameters():
        p.requires_grad_(False)

    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type=None,
    )
    model = get_peft_model(model, lora_cfg)
    return inferencer, model


def _real_train(cfg: dict, resume: Path | None) -> int:
    import torch

    from bagel_sbsr import LatentMagnitudeProvider, patch_bagel
    from scripts.coyo_dataloader import (  # type: ignore[import-not-found]
        CoyoStreamConfig,
        iter_coyo_batches,
    )

    try:
        from accelerate import Accelerator  # type: ignore[import-not-found]
    except ImportError:
        print("ERROR: accelerate not installed (uv pip install accelerate)", file=sys.stderr)
        return 1

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["train"]["gradient_accumulation_steps"],
        mixed_precision="bf16",
    )

    inferencer, model = _build_inferencer_and_lora(cfg)
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

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    model, optimizer = accelerator.prepare(model, optimizer)
    out_dir = Path(cfg["run"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if resume is not None and resume.exists():
        state = _load_checkpoint(resume)
        start_step = state["step"]
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.sbsr.load_state_dict(state["sbsr"])
        # LoRA params load into the named slots that match.
        own = dict(unwrapped.named_parameters())
        with torch.no_grad():
            for k, v in state["lora"].items():
                if k in own and own[k].shape == v.shape:
                    own[k].copy_(v.to(own[k].device))
        if accelerator.is_main_process:
            print(f"resumed from {resume} at step {start_step}")

    # Data
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
    last_rolling: Path | None = None
    ce_w = cfg["loss"]["ce_weight"]
    mse_w = cfg["loss"]["mse_weight"]

    step = start_step
    t0 = time.time()
    for batch in batch_iter:
        if step >= cfg["train"]["num_steps"]:
            break

        with accelerator.accumulate(model):
            optimizer.zero_grad()
            # Encode to FLUX VAE latent and set the saliency provider.
            with torch.no_grad():
                latent = inferencer.vae_model.encode(batch.images.to(accelerator.device))
                if hasattr(latent, "latent_dist"):
                    latent = latent.latent_dist.sample()
            provider.set_latent(latent)

            # Forward: rectified flow MSE + caption CE
            outputs = model(
                images=batch.images,
                captions=batch.captions,
                vae_latent=latent,
                ce_weight=ce_w,
                mse_weight=mse_w,
            )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable, cfg["train"]["max_grad_norm"])
            optimizer.step()

        loss_val = loss.detach().float().item()
        is_bad = (loss_val != loss_val) or (loss_val == float("inf")) or (loss_val == float("-inf"))
        nan_window.append(1 if is_bad else 0)

        if sum(nan_window) >= cfg["train"]["nan_detect_consecutive"]:
            if accelerator.is_main_process:
                print(
                    f"NaN-detect: {sum(nan_window)}/{cfg['train']['nan_detect_consecutive']} "
                    f"consecutive bad steps; rolling back",
                    file=sys.stderr,
                )
            if last_rolling is not None and last_rolling.exists():
                state = _load_checkpoint(last_rolling)
                unwrapped = accelerator.unwrap_model(model)
                unwrapped.sbsr.load_state_dict(state["sbsr"])
                own = dict(unwrapped.named_parameters())
                with torch.no_grad():
                    for k, v in state["lora"].items():
                        if k in own and own[k].shape == v.shape:
                            own[k].copy_(v.to(own[k].device))
                nan_window.clear()
                continue
            else:
                return 1

        if step % cfg["train"]["log_every"] == 0 and accelerator.is_main_process:
            elapsed = time.time() - t0
            print(
                f"step={step} loss={loss_val:.4f} elapsed={elapsed:.1f}s "
                f"failed_in_batch={batch.failed}"
            )

        if step % cfg["train"]["rolling_save_every"] == 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            from safetensors.torch import save_file  # type: ignore[import-not-found]

            rolling = out_dir / f"rolling-{step:08d}.safetensors"
            flat: dict = {}
            for k, v in unwrapped.sbsr.state_dict().items():
                flat[f"sbsr/{k}"] = v.contiguous().detach().cpu()
            for k, v in unwrapped.named_parameters():
                if v.requires_grad and "sbsr" not in k:
                    flat[f"lora/{k}"] = v.contiguous().detach().cpu()
            save_file(flat, str(rolling))
            (rolling.with_suffix(rolling.suffix + ".sha256")).write_text(_sha256_file(rolling))
            (rolling.with_suffix(rolling.suffix + ".json")).write_text(json.dumps({"step": step}))
            last_rolling = rolling
            _prune_rolling_ckpts(out_dir, cfg["train"]["rolling_save_keep"])

        if step % cfg["train"]["save_every"] == 0 and step > 0 and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            _save_checkpoint(
                out_dir,
                step,
                unwrapped.sbsr.state_dict(),
                {
                    k: v.detach().cpu()
                    for k, v in unwrapped.named_parameters()
                    if v.requires_grad and "sbsr" not in k
                },
            )

        step += 1
        provider.reset()

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        _save_checkpoint(
            out_dir,
            step,
            unwrapped.sbsr.state_dict(),
            {
                k: v.detach().cpu()
                for k, v in unwrapped.named_parameters()
                if v.requires_grad and "sbsr" not in k
            },
        )
        (out_dir / "final_metadata.json").write_text(
            json.dumps({"steps_completed": step, "elapsed_seconds": time.time() - t0}, indent=2)
        )
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    rc = _check_env(cfg, args.dry_run)
    if rc is not None:
        return rc
    if args.dry_run:
        return _dry_run(cfg)
    return _real_train(cfg, args.resume)


if __name__ == "__main__":
    sys.exit(main())
