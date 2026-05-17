"""Diffusers-compatible pipeline wrapper for BAGEL-SBSR.

This is a thin adapter that follows the Diffusers `DiffusionPipeline` API
contract (`__call__` returning an `ImagePipelineOutput`-shaped dict) while
delegating the heavy lifting to BAGEL's `InterleaveInferencer`. It does
*not* re-implement the rectified-flow sampler — that lives in the upstream
`InterleaveInferencer.gen_image` method.

Usage:
    from bagel_sbsr.pipeline_bagel_sbsr import BagelSBSRPipeline
    pipe = BagelSBSRPipeline.from_local(
        weights_dir="weights/bagel-7b-mot",
        vendor_dir="vendor/bagel-upstream",
        ckpt_path="runs/s3/ckpt-00020000.safetensors",
    )
    out = pipe("a calico cat on a windowsill", num_inference_steps=4)
    out.images[0].save("cat.png")
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BagelSBSRPipelineOutput:
    images: list = field(default_factory=list)
    text: str | None = None


class BagelSBSRPipeline:
    """Diffusers-shaped wrapper around BAGEL InterleaveInferencer + SBSR."""

    def __init__(self, inferencer: Any, provider: Any) -> None:
        self.inferencer = inferencer
        self.provider = provider

    @classmethod
    def from_local(
        cls,
        weights_dir: str | Path,
        vendor_dir: str | Path = "vendor/bagel-upstream",
        *,
        ckpt_path: str | Path | None = None,
        lambda_init: float = 0.5,
        mu_init: float = 0.25,
        top_k: int = 64,
    ) -> BagelSBSRPipeline:
        weights_dir = Path(weights_dir)
        vendor_dir = Path(vendor_dir).resolve()
        if str(vendor_dir) not in sys.path:
            sys.path.insert(0, str(vendor_dir))

        scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from sanity_inference import _build_inferencer  # type: ignore[import-not-found]

        from .hook import patch_bagel
        from .latent_saliency import LatentMagnitudeProvider

        inferencer = _build_inferencer(weights_dir)
        provider = LatentMagnitudeProvider()
        patch_bagel(
            inferencer.model,
            lambda_init=lambda_init,
            mu_init=mu_init,
            top_k=top_k,
            saliency_provider=provider,
            learnable=False,
        )
        if ckpt_path is not None:
            cls._load_ckpt(inferencer.model, Path(ckpt_path))
        return cls(inferencer, provider)

    @staticmethod
    def _load_ckpt(model: Any, ckpt: Path) -> None:
        import torch
        from safetensors.torch import load_file  # type: ignore[import-not-found]

        flat = load_file(str(ckpt))
        own = dict(model.named_parameters())
        sbsr = getattr(model, "sbsr", None)
        sbsr_params = dict(sbsr.named_parameters()) if sbsr is not None else {}
        with torch.no_grad():
            for k, v in flat.items():
                if k.startswith("sbsr/") and sbsr is not None:
                    local = k[len("sbsr/") :]
                    if local in sbsr_params:
                        sbsr.get_parameter(local).copy_(v.to(sbsr_params[local].device))
                elif k.startswith("gen/"):
                    local = k[len("gen/") :]
                    if local in own and own[local].shape == v.shape:
                        own[local].copy_(v.to(own[local].device))
                elif k.startswith("lora/"):
                    local = k[len("lora/") :]
                    if local in own and own[local].shape == v.shape:
                        own[local].copy_(v.to(own[local].device))

    def __call__(
        self,
        prompt: str | None = None,
        *,
        image=None,
        num_inference_steps: int = 4,
        height: int = 512,
        width: int = 512,
        understanding: bool = False,
        max_think_tokens: int = 64,
    ) -> BagelSBSRPipelineOutput:
        self.provider.reset()
        if image is not None or understanding:
            out = self.inferencer(
                image=image,
                understanding_output=True,
                max_think_token_n=max_think_tokens,
            )
            return BagelSBSRPipelineOutput(images=[], text=out.get("text"))
        out = self.inferencer(
            text=prompt,
            num_timesteps=num_inference_steps,
            image_shapes=(height, width),
        )
        return BagelSBSRPipelineOutput(images=[out.get("image")], text=None)
