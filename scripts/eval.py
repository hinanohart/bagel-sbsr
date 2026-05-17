"""Evaluation pipeline — FID / CLIP-T / GenEval / T2I-CompBench / HPSv2.1.

Each metric is wrapped in its own callable so missing dependencies degrade
gracefully (the metric is reported as None and a JSON note records the
reason). Output:
    runs/eval/<run_name>/scores.json
    runs/eval/<run_name>/samples/<prompt_id>.png

Usage:
    uv run scripts/eval.py --config configs/eval.yaml --ckpt runs/s3/ckpt-XXXX.safetensors
    uv run scripts/eval.py --config configs/eval.yaml --dry-run

Reference implementations pinned in docs/EVAL.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument(
        "--metrics",
        nargs="*",
        default=["fid", "clip_t", "geneval", "t2i_compbench", "hpsv2"],
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _check_env(dry_run: bool) -> int | None:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP: torch missing", file=sys.stderr)
        return 77
    if dry_run:
        return None
    return None


# ----- metric kernels (each returns (score | None, note: str)) ----------------


def metric_fid(samples_dir: Path, ref_dir: Path, batch_size: int = 8) -> tuple[float | None, str]:
    try:
        from cleanfid import fid  # type: ignore[import-not-found]
    except ImportError:
        return None, "cleanfid not installed"
    if not samples_dir.exists() or not ref_dir.exists():
        return None, f"missing {samples_dir} or {ref_dir}"
    score = float(fid.compute_fid(str(samples_dir), str(ref_dir), batch_size=batch_size))
    return score, "ok"


def metric_clip_t(
    samples_dir: Path, prompts: list[str], model_id: str = "ViT-L-14"
) -> tuple[float | None, str]:
    try:
        import open_clip  # type: ignore[import-not-found]
        import torch
        from PIL import Image
    except ImportError as e:
        return None, f"deps missing ({type(e).__name__})"
    if not samples_dir.exists():
        return None, f"missing {samples_dir}"

    model, _, preprocess = open_clip.create_model_and_transforms(model_id, pretrained="openai")
    tokenizer = open_clip.get_tokenizer(model_id)
    model.eval()

    images = sorted(samples_dir.glob("*.png"))
    if not images:
        return None, "no images"
    scores = []
    with torch.no_grad():
        for img_path, prompt in zip(images, prompts, strict=False):
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0)
            text = tokenizer([prompt])
            img_feat = model.encode_image(img)
            txt_feat = model.encode_text(text)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            scores.append(float((img_feat @ txt_feat.T).item()))
    return sum(scores) / len(scores), "ok"


def metric_geneval(samples_dir: Path, refs: Path) -> tuple[float | None, str]:
    try:
        import importlib

        spec = importlib.util.find_spec("geneval")
        if spec is None:
            return None, "geneval not installed (see docs/EVAL.md for repo pin)"
    except Exception as e:
        return None, type(e).__name__
    return None, "wire to djghosh13/geneval CLI: `geneval eval --images ... --output ...`"


def metric_t2i_compbench(samples_dir: Path, refs: Path) -> tuple[float | None, str]:
    try:
        import importlib

        spec = importlib.util.find_spec("t2i_compbench")
        if spec is None:
            return None, "T2I-CompBench not installed (see docs/EVAL.md)"
    except Exception as e:
        return None, type(e).__name__
    return None, "wire to Karine-Huang/T2I-CompBench evaluator"


def metric_hpsv2(samples_dir: Path, prompts: list[str]) -> tuple[float | None, str]:
    try:
        import importlib

        spec = importlib.util.find_spec("hpsv2")
        if spec is None:
            return None, "hpsv2 not installed (uv pip install hpsv2)"
        import hpsv2  # type: ignore[import-not-found]
    except Exception as e:
        return None, type(e).__name__
    images = sorted(samples_dir.glob("*.png"))
    if not images:
        return None, "no images"
    scores = []
    for p, prompt in zip(images, prompts, strict=False):
        s = hpsv2.score([str(p)], prompt)[0]
        scores.append(float(s))
    return sum(scores) / len(scores), "ok"


# ----- main orchestration -----------------------------------------------------


def _generate_samples(cfg: dict, ckpt: Path | None, out_dir: Path) -> list[str]:
    """Generate evaluation images from the checkpoint. Returns the prompt list.

    In --dry-run we emit a placeholder PNG per prompt so downstream metrics
    can be exercised on synthetic data.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = cfg["prompts"]
    if ckpt is None:
        # Synthetic: produce 64x64 random-noise PNGs as stand-ins.
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return prompts
        for i, _ in enumerate(prompts):
            arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
            Image.fromarray(arr).save(out_dir / f"sample_{i:04d}.png")
        return prompts
    # Real path: load the checkpoint, install SBSR, sample via the InterleaveInferencer.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sanity_inference import _build_inferencer  # type: ignore[import-not-found]

    from bagel_sbsr import LatentMagnitudeProvider, patch_bagel

    inferencer = _build_inferencer(Path(cfg["model"]["weights_dir"]))
    provider = LatentMagnitudeProvider()
    patch_bagel(
        inferencer.model,
        lambda_init=0.5,
        mu_init=0.25,
        top_k=64,
        saliency_provider=provider,
        learnable=False,
    )
    if ckpt is not None:
        import torch
        from safetensors.torch import load_file  # type: ignore[import-not-found]

        flat = load_file(str(ckpt))
        own = dict(inferencer.model.named_parameters())
        sbsr = getattr(inferencer.model, "sbsr", None)
        sbsr_params = dict(sbsr.named_parameters()) if sbsr is not None else {}
        with torch.no_grad():
            for k, v in flat.items():
                if k.startswith("sbsr/") and sbsr is not None:
                    local = k[len("sbsr/") :]
                    if local in sbsr_params:
                        sbsr.get_parameter(local).copy_(
                            v.to(sbsr_params[local].dtype).to(sbsr_params[local].device)
                        )
                elif k.startswith("gen/"):
                    local = k[len("gen/") :]
                    if local in own and own[local].shape == v.shape:
                        own[local].copy_(v.to(own[local].dtype).to(own[local].device))
                elif k.startswith("lora/"):
                    local = k[len("lora/") :]
                    if local in own and own[local].shape == v.shape:
                        own[local].copy_(v.to(own[local].dtype).to(own[local].device))

    for i, prompt in enumerate(prompts):
        out = inferencer(
            text=prompt,
            num_timesteps=cfg.get("num_timesteps", 4),
            image_shapes=(cfg.get("height", 512), cfg.get("width", 512)),
        )
        img = out.get("image")
        if img is not None:
            img.save(out_dir / f"sample_{i:04d}.png")
    return prompts


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    rc = _check_env(args.dry_run)
    if rc is not None:
        return rc

    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    samples_dir = run_dir / "samples"
    ref_dir = Path(cfg["reference_set"]) if cfg.get("reference_set") else samples_dir

    prompts = _generate_samples(cfg, None if args.dry_run else args.ckpt, samples_dir)

    metric_callables: dict[str, Callable[[], tuple[Any, str]]] = {
        "fid": lambda: metric_fid(samples_dir, ref_dir),
        "clip_t": lambda: metric_clip_t(samples_dir, prompts),
        "geneval": lambda: metric_geneval(samples_dir, ref_dir),
        "t2i_compbench": lambda: metric_t2i_compbench(samples_dir, ref_dir),
        "hpsv2": lambda: metric_hpsv2(samples_dir, prompts),
    }

    scores: dict[str, dict] = {}
    for m in args.metrics:
        if m not in metric_callables:
            scores[m] = {"score": None, "note": f"unknown metric {m}"}
            continue
        score, note = metric_callables[m]()
        scores[m] = {"score": score, "note": note}
        print(f"{m}: {score!r} ({note})")

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    print(f"wrote {run_dir / 'scores.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
