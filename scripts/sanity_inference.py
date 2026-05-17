"""End-to-end smoke for BAGEL-7B-MoT: one T2I + one I2T pass via InterleaveInferencer.

Mirrors the upstream `inference.ipynb` setup (Apache-2.0, credited).

Exit codes (used by CI):
   0 — success
  77 — environment incomplete (no GPU / no weights / no vendor) — CI treats as skip
   1 — actual failure
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

WEIGHTS_DEFAULT = Path("weights/bagel-7b-mot")
VENDOR_DEFAULT = Path("vendor/bagel-upstream")
OUT_DEFAULT = Path("outputs/sanity")
PROMPT_DEFAULT = "a photograph of an astronaut riding a horse on the moon, cinematic"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--weights", type=Path, default=WEIGHTS_DEFAULT)
    p.add_argument("--vendor", type=Path, default=VENDOR_DEFAULT)
    p.add_argument("--out", type=Path, default=OUT_DEFAULT)
    p.add_argument("--prompt", type=str, default=PROMPT_DEFAULT)
    p.add_argument("--nfe", type=int, default=4, help="num_timesteps passed to gen_image")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _check_env(args: argparse.Namespace) -> int | None:
    if not args.weights.exists() or not (args.weights / "config.json").exists():
        print(
            f"SKIP: weights missing at {args.weights}; run scripts/download_bagel.py first",
            file=sys.stderr,
        )
        return 77
    if not args.vendor.exists() or not (args.vendor / ".git").exists():
        print(
            f"SKIP: vendor source missing at {args.vendor}; run scripts/install_bagel_src.sh first",
            file=sys.stderr,
        )
        return 77
    try:
        import torch
    except ImportError:
        print("SKIP: torch not installed", file=sys.stderr)
        return 77
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device", file=sys.stderr)
        return 77
    return None


def _add_vendor(vendor: Path) -> None:
    vp = str(vendor.resolve())
    if vp not in sys.path:
        sys.path.insert(0, vp)


def _build_inferencer(model_path: Path):
    """Build InterleaveInferencer following upstream inference.ipynb setup."""
    import torch
    from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    from data.data_utils import add_special_tokens  # type: ignore[import-not-found]
    from data.transforms import ImageTransform  # type: ignore[import-not-found]
    from inferencer import InterleaveInferencer  # type: ignore[import-not-found]
    from modeling.autoencoder import load_ae  # type: ignore[import-not-found]
    from modeling.bagel import (  # type: ignore[import-not-found]
        Bagel,
        BagelConfig,
        Qwen2Config,
        Qwen2ForCausalLM,
        SiglipVisionConfig,
        SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer  # type: ignore[import-not-found]

    mp = str(model_path)

    llm_config = Qwen2Config.from_json_file(os.path.join(mp, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(mp, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(mp, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(mp)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(mp, "ema.safetensors"),
        device_map="auto",
        offload_folder=None,
        dtype=torch.bfloat16,
        force_hooks=True,
    ).eval()
    vae_model = vae_model.to("cuda").to(torch.bfloat16).eval()

    return InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )


def main() -> int:
    args = parse_args()
    rc = _check_env(args)
    if rc is not None:
        return rc

    _add_vendor(args.vendor)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    try:
        inferencer = _build_inferencer(args.weights)
    except Exception as e:
        print(f"ERROR: failed to build inferencer ({type(e).__name__}: {e})", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)

    print(f"T2I @ nfe={args.nfe}, prompt={args.prompt!r}")
    out = inferencer(
        text=args.prompt,
        num_timesteps=args.nfe,
        image_shapes=(args.height, args.width),
    )
    image = out.get("image")
    if image is None:
        print("ERROR: T2I returned no image", file=sys.stderr)
        return 1
    img_path = args.out / "t2i.png"
    image.save(img_path)
    print(f"  -> {img_path}")

    print(f"I2T on {img_path}")
    out = inferencer(image=image, understanding_output=True, max_think_token_n=64)
    caption = out.get("text") or ""
    cap_path = args.out / "i2t.txt"
    cap_path.write_text(caption.strip() + "\n")
    print(f"  -> {cap_path}: {caption.strip()!r}")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
