"""Launch a BAGEL-SBSR training job on RunPod.

Reads `RUNPOD_API_KEY` and `HF_TOKEN` from the local environment, forwards
their *values* into the pod environment via the RunPod API. Tokens never
appear in the pod entrypoint command, never in stdout, and exceptions are
caught and re-emitted with type-only messages so traceback echoes cannot
leak the payload.

Usage:
    RUNPOD_API_KEY=... HF_TOKEN=... uv run scripts/launch_runpod.py --config configs/s1.yaml

Exit codes:
    0  success (training finished)
   77  environment incomplete (no RUNPOD_API_KEY, no runpod SDK)
    1  failure (pod terminated, budget exceeded, etc.)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--script", type=Path, default=Path("scripts/train_s1.py"))
    p.add_argument("--poll-seconds", type=int, default=300)
    p.add_argument(
        "--image",
        default="runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04",
    )
    p.add_argument("--dry-run", action="store_true", help="print the pod spec but do not launch")
    return p.parse_args()


def _safe_str(e: BaseException) -> str:
    """Return a non-leaking string for an exception (type name only)."""
    return type(e).__name__


def main() -> int:
    args = parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    api_key = os.environ.get("RUNPOD_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        print("SKIP: RUNPOD_API_KEY not set", file=sys.stderr)
        return 77
    if not hf_token:
        print("SKIP: HF_TOKEN not set", file=sys.stderr)
        return 77

    # Pod env is uploaded by the RunPod SDK; the values stay server-side.
    # We pass the actual secrets through `env`, not through the entrypoint
    # command, so they never appear in process listings.
    pod_env = {
        "HF_TOKEN": hf_token,
        "WANDB_API_KEY": wandb_key,
        "PYTHONPATH": "vendor/bagel-upstream:/workspace/src",
    }

    entrypoint_cmd = [
        "bash",
        "-lc",
        (
            "set -euo pipefail; "
            "scripts/install_bagel_src.sh; "
            "uv run scripts/download_bagel.py --dest weights/bagel-7b-mot; "
            f"accelerate launch {args.script} --config {args.config}"
        ),
    ]

    spec_for_log = {
        "name": cfg["run"]["name"],
        "image": args.image,
        "gpu_type": cfg["cluster"]["gpu_type"],
        "gpu_count": cfg["cluster"]["gpus"],
        "spot": cfg["cluster"].get("spot", True),
        "env_keys": sorted(pod_env.keys()),
        "entrypoint": entrypoint_cmd,
        "max_runtime_hours": cfg["cluster"]["max_runtime_hours"],
        "max_cost_usd": min(
            cfg["cluster"]["max_cost_usd"], cfg["safety"]["budget_hard_ceiling_usd"]
        ),
    }

    print("Pod spec (secrets redacted):")
    import json as _json

    print(_json.dumps(spec_for_log, indent=2))

    if args.dry_run:
        print("DRY-RUN: not launching")
        return 0

    try:
        import runpod  # type: ignore[import-not-found]
    except ImportError:
        print("SKIP: runpod SDK not installed (uv pip install runpod)", file=sys.stderr)
        return 77

    runpod.api_key = api_key
    spec_for_api = {
        "name": cfg["run"]["name"],
        "image_name": args.image,
        "gpu_type_id": cfg["cluster"]["gpu_type"],
        "gpu_count": cfg["cluster"]["gpus"],
        "env": pod_env,
        "docker_args": " ".join(entrypoint_cmd[2:]) if len(entrypoint_cmd) >= 3 else "",
    }

    try:
        pod = runpod.create_pod(**spec_for_api)
    except Exception as e:
        print(f"ERROR: pod creation failed ({_safe_str(e)})", file=sys.stderr)
        return 1

    pod_id = pod.get("id") if isinstance(pod, dict) else None
    print(f"Pod launched: {pod_id}")

    while True:
        time.sleep(args.poll_seconds)
        try:
            status = runpod.get_pod(pod_id)
        except Exception as e:
            print(f"poll error ({_safe_str(e)}); continuing", file=sys.stderr)
            continue
        state = status.get("desiredStatus") if isinstance(status, dict) else None
        runtime = status.get("runtime", {}) if isinstance(status, dict) else {}
        print(f"  status={state} cost=${runtime.get('costPerHr', '?')}")
        if state in ("EXITED", "TERMINATED", "FAILED"):
            return 0 if state == "EXITED" else 1


if __name__ == "__main__":
    sys.exit(main())
