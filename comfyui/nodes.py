"""ComfyUI custom nodes for BAGEL-SBSR.

Install: drop this directory under `ComfyUI/custom_nodes/bagel_sbsr/`. ComfyUI
auto-discovers `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` at
startup.

Nodes:
  BagelSBSRLoader   — load BAGEL-7B-MoT + SBSR module, optional checkpoint
  BagelSBSRSampler  — text -> image (T2I) and image -> text (I2T)
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))


class BagelSBSRLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weights_dir": ("STRING", {"default": "weights/bagel-7b-mot"}),
                "vendor_dir": ("STRING", {"default": "vendor/bagel-upstream"}),
                "ckpt_path": ("STRING", {"default": ""}),
                "lambda_init": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5.0, "step": 0.01}),
                "mu_init": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 5.0, "step": 0.01}),
                "top_k": ("INT", {"default": 64, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("BAGEL_SBSR_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "BAGEL-SBSR"

    def load(self, weights_dir, vendor_dir, ckpt_path, lambda_init, mu_init, top_k):
        sys.path.insert(0, str(Path(vendor_dir).resolve()))
        from bagel_sbsr import LatentMagnitudeProvider, patch_bagel

        scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        sys.path.insert(0, str(scripts_dir))
        from sanity_inference import _build_inferencer  # type: ignore[import-not-found]

        inferencer = _build_inferencer(Path(weights_dir))
        provider = LatentMagnitudeProvider()
        patch_bagel(
            inferencer.model,
            lambda_init=lambda_init,
            mu_init=mu_init,
            top_k=top_k,
            saliency_provider=provider,
            learnable=False,
        )

        if ckpt_path:
            import torch
            from safetensors.torch import load_file

            flat = load_file(ckpt_path)
            own = dict(inferencer.model.named_parameters())
            with torch.no_grad():
                for k, v in flat.items():
                    if k.startswith("gen/"):
                        local = k[4:]
                        if local in own and own[local].shape == v.shape:
                            own[local].copy_(v.to(own[local].device))

        return ({"inferencer": inferencer, "provider": provider},)


class BagelSBSRSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("BAGEL_SBSR_MODEL",),
                "prompt": (
                    "STRING",
                    {"multiline": True, "default": "a calico cat on a windowsill"},
                ),
                "num_steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "mode": (["t2i", "i2t"], {"default": "t2i"}),
            },
            "optional": {"image_in": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "sample"
    CATEGORY = "BAGEL-SBSR"

    def sample(self, model, prompt, num_steps, width, height, mode, image_in=None):
        import torch

        inferencer = model["inferencer"]
        provider = model["provider"]
        provider.reset()

        if mode == "t2i":
            out = inferencer(text=prompt, num_timesteps=num_steps, image_shapes=(height, width))
            img = out.get("image")
            if img is None:
                blank = torch.zeros(1, height, width, 3)
                return (blank, "")
            import numpy as np

            arr = np.array(img.convert("RGB")).astype("float32") / 255.0
            tensor = torch.from_numpy(arr)[None, ...]
            return (tensor, "")
        else:  # i2t
            from PIL import Image

            if image_in is None:
                return (torch.zeros(1, height, width, 3), "")
            import numpy as np

            arr = (image_in[0].cpu().numpy() * 255).clip(0, 255).astype("uint8")
            pil = Image.fromarray(arr)
            out = inferencer(image=pil, understanding_output=True, max_think_token_n=64)
            return (image_in, out.get("text") or "")


NODE_CLASS_MAPPINGS = {
    "BagelSBSRLoader": BagelSBSRLoader,
    "BagelSBSRSampler": BagelSBSRSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BagelSBSRLoader": "BAGEL-SBSR Loader",
    "BagelSBSRSampler": "BAGEL-SBSR Sampler",
}
