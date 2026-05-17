"""Small helpers for environment / dependency / device probing.

These exist so unit tests can introspect the environment without importing torch
or huggingface_hub at module load.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def has_hf_token() -> bool:
    """True iff HF_TOKEN env var is set and non-empty. Never returns the value."""
    return bool(os.environ.get("HF_TOKEN"))


def has_runpod_key() -> bool:
    """True iff RUNPOD_API_KEY env var is set and non-empty. Never returns the value."""
    return bool(os.environ.get("RUNPOD_API_KEY"))


def has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


def has_cuda() -> bool:
    if not has_torch():
        return False
    import torch

    return torch.cuda.is_available()


def weights_present(path: str | Path = "weights/bagel-7b-mot") -> bool:
    p = Path(path)
    return p.is_dir() and (p / "config.json").exists()


def vendor_present(path: str | Path = "vendor/bagel-upstream") -> bool:
    p = Path(path)
    return p.is_dir() and (p / ".git").exists()
