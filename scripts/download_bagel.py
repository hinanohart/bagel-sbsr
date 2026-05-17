"""Download BAGEL-7B-MoT weights via huggingface_hub snapshot_download.

Auth is taken from the HF_TOKEN environment variable. We never read or print the token.

Usage:
    HF_TOKEN=... uv run scripts/download_bagel.py --dest weights/bagel-7b-mot

Notes:
- The default destination is `weights/bagel-7b-mot/` (gitignored).
- Idempotent: re-runs skip already-downloaded files.
- `--check-only` exits 0 if a complete snapshot exists, 1 otherwise.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ID = "ByteDance-Seed/BAGEL-7B-MoT"
EXPECTED_FILES = (
    "config.json",
    "llm_config.json",
    "vit_config.json",
    "tokenizer_config.json",
    "ae.safetensors",
    "ema.safetensors",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dest", type=Path, default=Path("weights/bagel-7b-mot"))
    p.add_argument(
        "--check-only", action="store_true", help="exit 0 if snapshot looks complete, 1 otherwise"
    )
    p.add_argument("--allow-patterns", nargs="*", default=None, help="optional HF allow_patterns")
    p.add_argument("--revision", default=None, help="optional HF revision pin")
    return p.parse_args()


def is_complete(dest: Path) -> bool:
    return dest.is_dir() and all((dest / f).exists() for f in EXPECTED_FILES)


def main() -> int:
    args = parse_args()
    dest: Path = args.dest

    if args.check_only:
        ok = is_complete(dest)
        print(f"snapshot {'complete' if ok else 'incomplete'} at {dest}")
        return 0 if ok else 1

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN env var is not set; export it before running.", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed. Run: uv pip install huggingface_hub",
            file=sys.stderr,
        )
        return 3

    dest.mkdir(parents=True, exist_ok=True)

    kwargs: dict = {"repo_id": REPO_ID, "local_dir": str(dest), "token": token}
    if args.allow_patterns is not None:
        kwargs["allow_patterns"] = args.allow_patterns
    if args.revision is not None:
        kwargs["revision"] = args.revision

    print(f"Downloading {REPO_ID} -> {dest}")
    try:
        path = snapshot_download(**kwargs)
    except Exception as e:
        # Scrub the token from any exception text. We do NOT print the exception
        # message body — huggingface_hub <0.24 has been known to embed the
        # bearer token into URLs inside RepositoryNotFoundError messages.
        print(f"ERROR: snapshot_download failed ({type(e).__name__})", file=sys.stderr)
        return 5
    print(f"OK: {path}")

    if not is_complete(dest):
        print("WARNING: snapshot finished but expected files missing", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
