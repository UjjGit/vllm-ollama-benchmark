"""Offline Qwen2.5-VL-3B-Instruct inference with vLLM.

Runs inside WSL (Ubuntu-24.04). vLLM has no native Windows build.

Usage:
    python run_qwen_vl.py IMAGE [IMAGE ...] -p "What is in this image?"

IMAGE may be a local path (Windows paths like C:\\pics\\a.png are translated
to /mnt/c/... automatically) or an http(s) URL.
"""

import argparse
import io
import os
import re
import sys
import urllib.request

# --- WSL2 workarounds. Both must be set before vllm is imported. ---

# vLLM's v2 model runner allocates UVA buffers, which require pinned memory.
# On WSL2 vLLM disables pinned memory by default (small perf regression on old
# kernels) and the engine then dies with "UVA is not available". This kernel is
# 6.6.x, well past the 4.19.121 gate, so opting back in is safe.
os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

# FlashInfer's top-k/top-p sampler is JIT-compiled with nvcc at startup. Only
# the CUDA *driver* ships with WSL, not the toolkit, so the build fails with
# "Could not find nvcc". Fall back to vLLM's PyTorch-native sampler; the
# difference is immaterial at single-request batch sizes. Remove this if you
# ever install the full CUDA toolkit.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from PIL import Image

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# Vision-token budget per image. Qwen2.5-VL packs 28x28 pixels per token, then
# merges 2x2, so max_pixels=1280*28*28 caps an image at ~320 tokens. Raising
# this improves detail on dense images at the cost of KV cache and latency.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

_WIN_PATH = re.compile(r"^([A-Za-z]):[\\/](.*)$")


def load_image(src: str) -> Image.Image:
    """Load an image from a URL, POSIX path, or Windows-style path."""
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data))
    else:
        m = _WIN_PATH.match(src)
        if m:
            drive, rest = m.groups()
            src = f"/mnt/{drive.lower()}/{rest.replace(chr(92), '/')}"
        if not os.path.exists(src):
            sys.exit(f"error: image not found: {src}")
        img = Image.open(src)
    # vLLM's processor expects RGB; drop alpha/palette channels.
    return img.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen2.5-VL inference via vLLM")
    ap.add_argument("images", nargs="+", help="image paths or URLs")
    ap.add_argument("-p", "--prompt", default="Describe this image in detail.")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="context window; lower it if vLLM reports insufficient KV cache",
    )
    ap.add_argument(
        "--gpu-mem",
        type=float,
        default=0.85,
        help="fraction of the 16GB VRAM vLLM may claim",
    )
    args = ap.parse_args()

    images = [load_image(s) for s in args.images]

    # Imported here so --help stays fast (vLLM import takes ~10s).
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        limit_mm_per_prompt={"image": len(images)},
        mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS},
    )

    processor = AutoProcessor.from_pretrained(MODEL)
    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": args.prompt})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate(
        {
            "prompt": prompt,
            # A single image must be passed bare, not as a 1-element list.
            "multi_modal_data": {"image": images if len(images) > 1 else images[0]},
        },
        SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens),
    )

    print("\n" + "=" * 60)
    print(outputs[0].outputs[0].text.strip())
    print("=" * 60)


if __name__ == "__main__":
    main()
