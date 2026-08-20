"""
Extract content from a PDF using Ollama + Qwen2.5-VL-3B-Instruct.
Run alongside make_test_image.py (vLLM) to benchmark both backends.

Prerequisites:
    ollama serve                  # start the Ollama server (separate terminal)
    ollama pull qwen2.5vl:3b      # pull the model once

Usage:
    python run_ollama_vl.py                           # uses sample PDF from make_test_image.py
    python run_ollama_vl.py --pdf /path/to/doc.pdf    # use your own PDF

Install dependencies (once):
    uv pip install ollama --python /home/ujjwal/vllm/.venv/bin/python
"""

import argparse
import asyncio
import os
import sys
import time

OLLAMA_MODEL = "qwen2.5vl:3b"
SAMPLE_PDF   = "/tmp/sample_document.pdf"
OUTPUT_DIR   = "/mnt/c/UG/vLLM_Local/output"
OUTPUT_TXT   = os.path.join(OUTPUT_DIR, "extracted_text_ollama.txt")
PAGE_IMG_DIR = "/tmp/pdf_pages_ollama"

PROMPT_TEXT = (
    "Extract all text visible on this document page. "
    "Preserve headings, bullet points, and layout as closely as possible."
)


# ── PDF → images ───────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 150) -> list:
    import fitz  # pymupdf
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    paths = []
    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        path = os.path.join(out_dir, f"page_{i:02d}.png")
        img.save(path)
        print(f"  rendered page {i} -> {path}")
        paths.append(path)
    doc.close()
    return paths


# ── Ollama extraction (parallel) ───────────────────────────────────────────────

async def _warmup(client) -> float:
    """Load the model into VRAM with a tiny request; return seconds taken."""
    t0 = time.perf_counter()
    await client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        options={"num_predict": 1},
    )
    return time.perf_counter() - t0


async def _process_page(client, i: int, path: str) -> dict:
    """Send one page to Ollama and return results."""
    with open(path, "rb") as f:
        img_bytes = f.read()

    t_start = time.perf_counter()
    response = await client.chat(
        model=OLLAMA_MODEL,
        messages=[{
            "role": "user",
            "content": PROMPT_TEXT,
            "images": [img_bytes],
        }],
    )
    elapsed = time.perf_counter() - t_start

    text       = response["message"]["content"].strip()
    tokens_out = response.get("eval_count", 0)
    eval_ns    = response.get("eval_duration", 0)
    tps        = (tokens_out / eval_ns * 1e9) if eval_ns else 0.0
    return {
        "page":       i,
        "path":       path,
        "text":       text,
        "time_s":     round(elapsed, 3),
        "tokens_out": tokens_out,
        "tps":        round(tps, 1),
    }


async def _extract_parallel(page_paths: list, txt_path: str) -> tuple:
    import ollama

    client = ollama.AsyncClient()

    print(f"Loading model {OLLAMA_MODEL} into Ollama …")
    load_time = await _warmup(client)
    print(f"  model ready in {load_time:.2f}s")

    print(f"  sending all {len(page_paths)} pages in parallel …")
    t_wall_start = time.perf_counter()
    results = await asyncio.gather(
        *[_process_page(client, i, path) for i, path in enumerate(page_paths, 1)]
    )
    total_wall = time.perf_counter() - t_wall_start

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for r in results:
            header = (
                f"\n{'='*60}\n"
                f"  PAGE {r['page']}  -  {os.path.basename(r['path'])}"
                f"  ({r['time_s']}s | {r['tokens_out']} tok | {r['tps']} tok/s)\n"
                f"{'='*60}\n"
            )
            print(header + r["text"])
            f.write(header + r["text"] + "\n")
        f.write(f"\n{'='*60}\n")

    print(f"\nOutput saved -> {txt_path}")
    return load_time, total_wall, results


def extract_content(page_paths: list, txt_path: str) -> tuple:
    return asyncio.run(_extract_parallel(page_paths, txt_path))


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(load_time: float, total_wall: float, page_results: list) -> None:
    n          = len(page_results)
    sum_serial = sum(r["time_s"] for r in page_results)

    print("\n" + "=" * 60)
    print("  OLLAMA BENCHMARK SUMMARY  (parallel)")
    print("=" * 60)
    print(f"  Backend:              Ollama (parallel)")
    print(f"  Model:                {OLLAMA_MODEL}")
    print(f"  Pages processed:      {n}")
    print(f"  Model load time:      {load_time:.2f}s")
    print(f"  Total wall-clock:     {total_wall:.2f}s  <- actual time taken")
    print(f"  Sum of page times:    {sum_serial:.2f}s  (sequential equivalent)")
    print(f"  Parallelism speedup:  {sum_serial/total_wall:.1f}x")
    print()
    print(f"  {'Page':<6} {'Time (s)':>9} {'Tokens':>8} {'Tok/s':>8}")
    print(f"  {'-'*4}  {'-'*9}  {'-'*6}  {'-'*8}")
    for r in page_results:
        print(f"  {r['page']:<6} {r['time_s']:>9.2f} {r['tokens_out']:>8} {r['tps']:>8.1f}")
    print("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ollama Qwen2.5-VL-3B PDF extraction + benchmark"
    )
    ap.add_argument(
        "--pdf",
        default=None,
        help="path to PDF; omit to use the sample from make_test_image.py",
    )
    ap.add_argument("--dpi", type=int, default=150, help="render resolution (default 150)")
    args = ap.parse_args()

    pdf_path = args.pdf or SAMPLE_PDF
    if not os.path.exists(pdf_path):
        sys.exit(
            f"error: PDF not found: {pdf_path}\n"
            "Run make_test_image.py first to generate the sample PDF, "
            "or pass --pdf <path> to use your own."
        )

    print(f"Rendering PDF pages (dpi={args.dpi}) …")
    page_images = pdf_to_images(pdf_path, PAGE_IMG_DIR, dpi=args.dpi)
    print(f"  {len(page_images)} pages ready\n")

    load_time, total_wall, page_results = extract_content(page_images, OUTPUT_TXT)
    print_summary(load_time, total_wall, page_results)


if __name__ == "__main__":
    main()
