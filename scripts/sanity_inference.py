"""End-to-end smoke: load BAGEL-7B-MoT and run one T2I + one I2T pass.

This is *sanity*, not benchmarking. The script:
  1. checks that the weights directory exists and looks complete,
  2. checks that vendor/bagel-upstream/ has been installed,
  3. loads the model in bf16 on the first available CUDA device (or skips with code 77 if no GPU),
  4. runs one T2I generation at NFE=4 with a fixed prompt,
  5. runs one I2T caption pass on the just-generated image,
  6. writes both outputs to `outputs/sanity/` and exits 0.

Exit codes (used by CI):
   0 — success
  77 — environment incomplete (no GPU, no weights, no vendor source) — CI treats as skip
   1 — actual failure
"""

from __future__ import annotations

import argparse
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
    p.add_argument("--nfe", type=int, default=4)
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


def main() -> int:
    args = parse_args()
    rc = _check_env(args)
    if rc is not None:
        return rc

    _add_vendor(args.vendor)
    args.out.mkdir(parents=True, exist_ok=True)

    import torch

    # BAGEL upstream classes are imported here so import errors only surface when running
    # with a real environment, not at unit-test time.
    try:
        from modeling.bagel.qwen2_navit import Bagel  # type: ignore[import-not-found]
    except Exception as e:
        print(f"ERROR: failed to import BAGEL classes from vendor: {e}", file=sys.stderr)
        return 1

    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    print(f"Loading BAGEL from {args.weights} on {device} ({dtype}) ...")
    model = Bagel.from_pretrained(str(args.weights), torch_dtype=dtype).to(device)
    model.eval()

    torch.manual_seed(args.seed)

    # --- T2I ---
    print(f"T2I @ NFE={args.nfe}, prompt={args.prompt!r}")
    with torch.inference_mode():
        image = model.generate_image(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.nfe,
        )
    img_path = args.out / "t2i.png"
    image.save(img_path)
    print(f"  -> {img_path}")

    # --- I2T ---
    print(f"I2T on {img_path}")
    with torch.inference_mode():
        caption = model.caption(image, max_new_tokens=64)
    cap_path = args.out / "i2t.txt"
    cap_path.write_text(caption.strip() + "\n")
    print(f"  -> {cap_path}: {caption.strip()!r}")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
