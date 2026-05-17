"""P1 tests: download/install/sanity scripts can be imported and parse args.

Heavy paths (actual HF download, real GPU inference) are skipped when the
environment is incomplete — verified via _env helpers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


@pytest.mark.smoke
def test_env_helpers_importable():
    from bagel_sbsr import _env

    assert callable(_env.has_hf_token)
    assert callable(_env.has_runpod_key)
    assert isinstance(_env.has_torch(), bool)
    assert isinstance(_env.has_cuda(), bool)


@pytest.mark.smoke
def test_env_helpers_never_leak_secret_values(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "sentinel-should-not-be-returned")
    monkeypatch.setenv("RUNPOD_API_KEY", "sentinel-should-not-be-returned")
    from bagel_sbsr import _env

    assert _env.has_hf_token() is True
    assert _env.has_runpod_key() is True
    # public API surface must be boolean-only; verify no callable returns a string
    for name in ("has_hf_token", "has_runpod_key", "has_torch", "has_cuda"):
        val = getattr(_env, name)()
        assert isinstance(val, bool), f"{name} returned {type(val).__name__}, not bool"


@pytest.mark.smoke
def test_download_bagel_help_runs():
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "download_bagel.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert r.returncode == 0, r.stderr
    assert "HF_TOKEN" in r.stdout or "HF_TOKEN" in r.stderr or "ByteDance-Seed" in r.stdout


@pytest.mark.smoke
def test_download_bagel_check_only_returns_1_for_missing():
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "download_bagel.py"),
            "--check-only",
            "--dest",
            "/tmp/_p1_does_not_exist",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    # check_only must NOT require HF_TOKEN; returns 1 (incomplete) for a missing dir
    assert r.returncode == 1, f"got rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}"


@pytest.mark.smoke
def test_sanity_inference_help_runs():
    r = subprocess.run(
        [sys.executable, str(SCRIPTS / "sanity_inference.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert r.returncode == 0, r.stderr


@pytest.mark.smoke
def test_install_bagel_src_help_runs():
    r = subprocess.run(
        ["bash", str(SCRIPTS / "install_bagel_src.sh"), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert r.returncode == 0, r.stderr
    assert "BAGEL_REPO_URL" in r.stdout or "Clone the BAGEL" in r.stdout


@pytest.mark.smoke
def test_bagel_vendor_api_signature_if_present():
    """When vendor/bagel-upstream/ exists, verify the expected API symbols are importable.

    This guards against the relay-capture scenario where sanity_inference.py
    references symbols that don't exist in upstream BAGEL. If vendor is not
    installed (the usual CI case), the test skips cleanly.
    """
    vendor = REPO_ROOT / "vendor" / "bagel-upstream"
    if not (vendor / ".git").exists():
        pytest.skip(f"vendor source absent at {vendor}; run scripts/install_bagel_src.sh")

    # Add to sys.path only for this test, then drop.
    inserted = str(vendor.resolve())
    sys.path.insert(0, inserted)
    try:
        from inferencer import InterleaveInferencer  # type: ignore[import-not-found]  # noqa: F401
        from modeling.bagel import (  # type: ignore[import-not-found]  # noqa: F401
            Bagel,
            BagelConfig,
        )
    finally:
        if sys.path and sys.path[0] == inserted:
            sys.path.pop(0)
