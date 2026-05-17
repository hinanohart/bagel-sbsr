"""HF Space demo (Gradio) for BAGEL-SBSR.

Deploy:
    cp demo/app.py demo/requirements.txt  ->  HuggingFace Space (Gradio SDK)

Locally:
    uv run python demo/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make local src/ importable when running the demo without `pip install -e .`.
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))


def _build_pipe():
    """Build the pipeline lazily so missing GPU / weights surface to UI."""
    from bagel_sbsr.pipeline_bagel_sbsr import BagelSBSRPipeline

    weights_dir = os.environ.get("BAGEL_WEIGHTS_DIR", "weights/bagel-7b-mot")
    vendor_dir = os.environ.get("BAGEL_VENDOR_DIR", "vendor/bagel-upstream")
    ckpt = os.environ.get("BAGEL_SBSR_CKPT")
    return BagelSBSRPipeline.from_local(
        weights_dir=weights_dir,
        vendor_dir=vendor_dir,
        ckpt_path=ckpt if ckpt else None,
    )


def _generate(pipe, prompt: str, num_steps: int, height: int, width: int):
    if pipe is None:
        return None, "pipeline not available (set BAGEL_WEIGHTS_DIR + run on GPU)"
    out = pipe(prompt, num_inference_steps=num_steps, height=height, width=width)
    if not out.images:
        return None, "no image returned"
    return out.images[0], None


def _understand(pipe, image, max_tokens: int):
    if pipe is None:
        return "pipeline not available"
    if image is None:
        return "(no image)"
    out = pipe(image=image, understanding=True, max_think_tokens=max_tokens)
    return out.text or "(no text)"


def main() -> int:
    try:
        import gradio as gr
    except ImportError:
        print("gradio not installed — uv pip install gradio", file=sys.stderr)
        return 1

    try:
        pipe = _build_pipe()
        startup_error = None
    except Exception as e:
        pipe = None
        startup_error = type(e).__name__

    with gr.Blocks(title="BAGEL-SBSR demo") as demo:
        gr.Markdown(
            "# BAGEL-SBSR\nSaliency-biased Sparse Routing on BAGEL-7B-MoT, distilled to 2-4 NFE."
        )
        if startup_error is not None:
            gr.Markdown(f"_Startup notice: {startup_error}_")

        with gr.Tab("Text → Image"):
            prompt = gr.Textbox(label="Prompt", value="a calico cat on a windowsill at sunset")
            steps = gr.Slider(1, 50, value=4, step=1, label="NFE")
            h = gr.Slider(256, 1024, value=512, step=64, label="Height")
            w = gr.Slider(256, 1024, value=512, step=64, label="Width")
            out_img = gr.Image(label="Output", type="pil")
            out_msg = gr.Textbox(label="Note")
            btn = gr.Button("Generate")
            btn.click(
                lambda p, n, hh, ww: _generate(pipe, p, n, hh, ww),
                inputs=[prompt, steps, h, w],
                outputs=[out_img, out_msg],
            )

        with gr.Tab("Image → Text"):
            in_img = gr.Image(label="Input image", type="pil")
            max_tok = gr.Slider(16, 256, value=64, step=8, label="Max think tokens")
            out_txt = gr.Textbox(label="Caption")
            btn2 = gr.Button("Caption")
            btn2.click(
                lambda im, mt: _understand(pipe, im, mt),
                inputs=[in_img, max_tok],
                outputs=[out_txt],
            )

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
