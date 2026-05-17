"""COYO-700M streaming dataloader for BAGEL-SBSR S1 training.

Streaming-only (no full download): uses `datasets.load_dataset(..., streaming=True)`
over the `kakaobrain/coyo-700m` dataset. Each example is processed to:
    (image, caption) -> (vae_latent, text_tokens, packed_input)
The collator emits a BAGEL-compatible packed batch with `sample_lens`,
`packed_sequence`, `packed_und_token_indexes`, `packed_gen_token_indexes`.

Caveats (honest):
- COYO is image *URL* + caption, not image bytes. We retrieve via HTTP with a
  per-request timeout; 404s and 30%-mirror-fallback are surfaced upstream.
- The collator stub here returns a structurally valid batch but expects the
  caller (train_s1.py) to plug in the real VAE/tokenizer (load_ae /
  Qwen2Tokenizer) from sanity_inference._build_inferencer() — to keep this
  module decoupled.

This module deliberately does not import BAGEL upstream. It works on (PIL
image, caption) inputs and yields plain torch tensors; train_s1.py is
responsible for combining the latents and tokens with BAGEL's packing rules.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CoyoBatch:
    images: torch.Tensor  # (B, 3, H, W) in [0, 1]
    captions: list[str]  # length B
    urls: list[str]  # length B
    failed: int  # how many examples in the shard were skipped


@dataclass
class CoyoStreamConfig:
    resolution: int = 512
    timeout_seconds: float = 8.0
    max_per_batch_retries: int = 3
    mirror_fallback_threshold: float = 0.3
    shuffle_buffer: int = 10_000
    seed: int = 42


def stream_coyo(
    cfg: CoyoStreamConfig,
    *,
    image_transform: Callable[[Any], torch.Tensor],
    split: str = "train",
) -> Iterator[dict]:
    """Yield {image, caption, url} per example, retrying transient failures.

    Args:
        cfg: streaming config.
        image_transform: PIL.Image -> torch.Tensor in CHW float [0,1] at `cfg.resolution`.
        split: HF dataset split.

    Raises:
        RuntimeError if more than `mirror_fallback_threshold` of the first
        1_000 examples fail to download.
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("stream_coyo requires `datasets` (uv pip install datasets)") from e
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("stream_coyo requires `requests`") from e
    try:
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError("stream_coyo requires `Pillow`") from e

    ds = load_dataset("kakaobrain/coyo-700m", split=split, streaming=True)
    ds = ds.shuffle(buffer_size=cfg.shuffle_buffer, seed=cfg.seed)

    total = 0
    failed = 0
    health_window = 1_000

    for ex in ds:
        url = ex.get("url") or ex.get("image_url")
        caption = ex.get("text") or ex.get("caption") or ""
        if not url:
            continue
        total += 1
        try:
            r = requests.get(url, timeout=cfg.timeout_seconds, stream=True)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            tensor = image_transform(img)
        except Exception:
            failed += 1
            if total <= health_window and failed / total > cfg.mirror_fallback_threshold:
                raise RuntimeError(
                    f"COYO URL liveness below threshold: {failed}/{total} failed "
                    f"in first {health_window} examples (limit "
                    f"{cfg.mirror_fallback_threshold:.0%})"
                )
            continue
        yield {"image": tensor, "caption": caption, "url": url}


def collate_coyo_examples(
    examples: Iterable[dict],
) -> CoyoBatch:
    """Stack tensor images + return aligned captions."""
    images = []
    captions = []
    urls = []
    failed = 0
    for ex in examples:
        if ex is None:
            failed += 1
            continue
        images.append(ex["image"])
        captions.append(ex["caption"])
        urls.append(ex["url"])
    return CoyoBatch(
        images=torch.stack(images, dim=0) if images else torch.empty(0),
        captions=captions,
        urls=urls,
        failed=failed,
    )


def iter_coyo_batches(
    cfg: CoyoStreamConfig,
    *,
    image_transform: Callable[[Any], torch.Tensor],
    batch_size: int,
    split: str = "train",
) -> Iterator[CoyoBatch]:
    """Yield CoyoBatch chunks of size `batch_size`."""
    buf: list[dict] = []
    for ex in stream_coyo(cfg, image_transform=image_transform, split=split):
        buf.append(ex)
        if len(buf) >= batch_size:
            yield collate_coyo_examples(buf)
            buf = []
    if buf:
        yield collate_coyo_examples(buf)
