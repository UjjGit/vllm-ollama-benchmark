# vLLM vs Ollama Benchmark — Qwen2.5-VL-3B on WSL2

A hands-on benchmark comparing **vLLM** and **Ollama** for local vision-language model inference, using `Qwen2.5-VL-3B-Instruct` on a single consumer GPU (RTX 5080 Laptop, 16 GB VRAM).

**Task:** Extract text from every page of a multi-page PDF document using the model, in both sequential and parallel modes.

---

## Results

### Sequential (one page at a time)

| Backend | Model precision | Total inference | Avg per page |
|---|---|---|---|
| Ollama | Q4 GGUF | **73.0s** | **8.1s** |
| vLLM | bfloat16 | 93.8s | 10.4s |

Ollama wins here — its Q4 quantization generates tokens 3–4x faster per single request. vLLM's batching advantages are idle.

### Parallel (all pages sent simultaneously)

| Backend | Model precision | Wall-clock | Speedup |
|---|---|---|---|
| vLLM | AWQ (4-bit) | **33.5s** | **8.9x** |
| Ollama | Q4 GGUF | 60.9s | 5.0x |

vLLM wins decisively — **1.8x faster wall-clock**. All 9 pages are processed in a single continuous GPU batch. Ollama queues requests internally and processes them serially.

---

## Key Takeaways

- **Use Ollama** for single-user, single-request workloads — simpler setup, lower latency.
- **Use vLLM** for concurrent workloads — multi-user APIs, batch pipelines, document processors. Its PagedAttention + continuous batching turns N sequential requests into near-single-request latency.
- **Always match quantization when benchmarking** — comparing bfloat16 vs Q4 is not a fair fight; the precision difference dominates everything else.

---

## Environment

| Component | Version |
|---|---|
| GPU | ASUS ROG Zephyrus RTX 5080 Laptop (16 GB, Blackwell sm_120) |
| WSL distro | Ubuntu 24.04.4, kernel 6.6.87.2 |
| Python | 3.12.3 |
| venv | `~/vllm/.venv` (managed by `uv`) |
| vLLM | 0.27.1 |
| Ollama | latest |
| PyTorch | 2.13.0+cu130 |
| Model | `Qwen/Qwen2.5-VL-3B-Instruct` / `Qwen2.5-VL-3B-Instruct-AWQ` |

---

## Files

| File | Purpose |
|---|---|
| `make_test_image.py` | Generate sample PDF + extract pages in **parallel** via vLLM HTTP server |
| `run_ollama_vl.py` | Same pipeline via **Ollama** (parallel async) |
| `run_qwen_vl.py` | Simple single-image VLM inference via vLLM offline API |
| `check_gpu.py` | Verify CUDA, sm_120 support, bf16 matmul, free VRAM |

---

## Setup

### 1. Install dependencies (WSL Ubuntu-24.04)

```bash
sudo apt-get install -y build-essential python3-dev zstd
```

### 2. Create venv and install vLLM

```bash
uv venv ~/vllm/.venv
source ~/vllm/.venv/bin/activate
uv pip install vllm pymupdf fpdf2 ollama
```

### 3. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5vl:3b
```

---

## Running the benchmark

### Start the vLLM server (Terminal 1)

```bash
source ~/vllm/.venv/bin/activate
VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 \
vllm serve Qwen/Qwen2.5-VL-3B-Instruct-AWQ --dtype half \
  --max-model-len 8192 --gpu-memory-utilization 0.85
```

### Run benchmarks (Terminal 2)

```bash
source ~/vllm/.venv/bin/activate

# vLLM — generates sample PDF, renders pages, sends all in parallel
python /mnt/c/UG/vLLM_Local/make_test_image.py

# Ollama — same PDF, same pages, parallel async
python /mnt/c/UG/vLLM_Local/run_ollama_vl.py

# Use your own PDF
python /mnt/c/UG/vLLM_Local/make_test_image.py --pdf /path/to/doc.pdf
python /mnt/c/UG/vLLM_Local/run_ollama_vl.py   --pdf /path/to/doc.pdf
```

Results are saved to `output/extracted_text.txt` and `output/extracted_text_ollama.txt`.

---

## WSL2 Workarounds

Set before vLLM is imported (already in all scripts):

- **`VLLM_WSL2_ENABLE_PIN_MEMORY=1`** — re-enables pinned memory for UVA buffers; safe on kernels >= 4.19.121.
- **`VLLM_USE_FLASHINFER_SAMPLER=0`** — disables FlashInfer's nvcc-compiled sampler (WSL ships the CUDA driver but not the full toolkit).
