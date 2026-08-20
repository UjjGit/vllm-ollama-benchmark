"""
Create a sample 5-10 page PDF and extract its content using Qwen2.5-VL-3B-Instruct-AWQ
via a running vLLM server. All pages are sent in parallel to leverage vLLM batching.

Start the server first (in a separate terminal):
    VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 \\
    vllm serve Qwen/Qwen2.5-VL-3B-Instruct --dtype bfloat16 \\
      --max-model-len 8192 --gpu-memory-utilization 0.85

Usage:
    python make_test_image.py                          # generates sample PDF then extracts
    python make_test_image.py --pdf /path/to/doc.pdf   # use an existing PDF
"""

import argparse
import asyncio
import base64
import os
import sys
import time

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

SAMPLE_PDF = "/tmp/sample_document.pdf"
OUTPUT_DIR = "/mnt/c/UG/vLLM_Local/output"
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "extracted_text.txt")
PAGE_IMG_DIR = "/tmp/pdf_pages"

# ── PDF content ────────────────────────────────────────────────────────────────

DOCUMENT = [
    # (page_title, sections)
    # Each section: (heading, body_lines)
    (
        "Cover Page",
        [
            (None, [
                "",
                "",
                "ACME CORPORATION",
                "",
                "Annual Performance Report",
                "Fiscal Year 2025",
                "",
                "",
                "Prepared by: Strategy & Analytics Team",
                "Date: August 2025",
                "",
                "CONFIDENTIAL",
            ]),
        ],
    ),
    (
        "Executive Summary",
        [
            ("Executive Summary", [
                "This report presents a comprehensive overview of ACME Corporation's "
                "performance during fiscal year 2025. Despite challenging market "
                "conditions, the company achieved significant growth across all key "
                "business segments.",
            ]),
            ("Financial Highlights", [
                "Total revenue reached $4.2 billion, representing a 12% year-over-year "
                "increase. Operating income grew by 18% to $840 million, reflecting "
                "improved operational efficiency and disciplined cost management.",
            ]),
            ("Strategic Initiatives", [
                "The company successfully launched three major product lines in Q2 and "
                "Q3, capturing 8% additional market share in the enterprise software "
                "segment. Digital transformation efforts reduced operational costs by "
                "$120 million.",
            ]),
            ("Workforce & Culture", [
                "Headcount grew to 28,400 employees. Employee engagement scores reached "
                "an all-time high of 82%, supported by expanded learning programs and a "
                "new hybrid work policy introduced in Q1.",
            ]),
        ],
    ),
    (
        "Key Performance Metrics",
        [
            ("Revenue & Profitability", [
                "Total Revenue:           $4.2B    (+12% YoY)",
                "Gross Profit Margin:     38.5%    (+2.1 pts)",
                "Operating Income:        $840M    (+18% YoY)",
                "Net Income:              $612M    (+15% YoY)",
                "Earnings Per Share:      $3.42    (+14% YoY)",
            ]),
            ("Operational Metrics", [
                "Employee Count:          28,400   (+1,200 vs prior year)",
                "Customer Satisfaction:   4.6 / 5.0  CSAT score",
                "Net Promoter Score:      72       (Industry avg: 54)",
                "Platform Uptime:         99.97%   SLA met",
                "Support Tickets Closed:  94.3%    within SLA",
            ]),
        ],
    ),
    (
        "Project Highlights",
        [
            ("Project Atlas - Cloud Migration", [
                "Migrated 85% of on-premise workloads to AWS, reducing infrastructure "
                "costs by $45M annually. The migration completed two months ahead of "
                "the original schedule.",
            ]),
            ("Project Nova - AI Product Suite", [
                "Launched four AI-powered modules integrated into the core platform. "
                "Early adoption by 1,200 enterprise clients generated $90M in "
                "incremental ARR.",
            ]),
            ("Project Horizon - Market Expansion", [
                "Entered five new international markets: Brazil, India, South Korea, "
                "UAE, and South Africa. Combined first-year revenue: $210M.",
            ]),
            ("Sustainability", [
                "Achieved carbon-neutral status for all US data centers. Renewable "
                "energy now powers 72% of global operations, up from 48% in 2024.",
            ]),
        ],
    ),
    (
        "Regional Performance",
        [
            ("North America", [
                "Revenue: $1.9B (+10% YoY). The US market remained the largest revenue "
                "contributor. Canada saw accelerated growth of 18%, driven by public "
                "sector contracts signed in Q2.",
            ]),
            ("Europe", [
                "Revenue: $1.1B (+14% YoY). Strong performance in the DACH region and "
                "the UK. GDPR compliance upgrades completed across all product lines "
                "opened new enterprise accounts in Germany and France.",
            ]),
            ("Asia-Pacific", [
                "Revenue: $0.8B (+16% YoY). India and South Korea were standout "
                "performers following the Project Horizon expansion. Japan and "
                "Australia maintained steady double-digit growth.",
            ]),
            ("Emerging Markets", [
                "Revenue: $0.4B (+31% YoY). Brazil and UAE generated $210M combined. "
                "South Africa showed promising early traction with $45M in H2 alone.",
            ]),
        ],
    ),
    (
        "Financial Statements Overview",
        [
            ("Income Statement ($ millions)", [
                "Revenue:                   4,200",
                "Cost of Revenue:           2,583",
                "Gross Profit:              1,617",
                "Operating Expenses:          777",
                "Operating Income:            840",
                "Interest & Other:            (48)",
                "Pre-tax Income:              792",
                "Income Tax (22.7%):         (180)",
                "Net Income:                  612",
            ]),
            ("Balance Sheet Highlights", [
                "Total Assets:   $6.8B   |   Total Liabilities: $2.9B",
                "Equity:         $3.9B   |   Cash & Equivalents: $1.1B",
                "Long-term Debt: $1.4B   |   Current Ratio: 2.1x",
            ]),
        ],
    ),
    (
        "Technology & Innovation",
        [
            ("R&D Investment", [
                "ACME invested $420M in R&D in FY2025, representing 10% of total "
                "revenue. Key focus areas included large language model tooling, "
                "real-time analytics, and edge computing infrastructure.",
            ]),
            ("Patents & IP", [
                "Filed 142 new patents in FY2025, bringing the total active portfolio "
                "to 1,840 patents across 34 countries. Three patents were licenced to "
                "industry partners generating $12M in royalty income.",
            ]),
            ("Engineering Headcount", [
                "Engineering & Product grew to 9,800 employees (+18% YoY), representing "
                "34% of total headcount. Two new engineering hubs opened in Bangalore "
                "and Warsaw.",
            ]),
        ],
    ),
    (
        "ESG & Corporate Responsibility",
        [
            ("Environmental", [
                "Carbon Emissions (Scope 1+2): -31% vs 2022 baseline.",
                "Renewable Energy Share: 72% of global operations.",
                "Water Usage Reduction: 18% through data center efficiency programs.",
                "Zero-waste-to-landfill: achieved at 12 of 15 office locations.",
            ]),
            ("Social", [
                "Diversity in leadership: 44% women and underrepresented groups.",
                "Community investment: $28M donated to STEM education programs.",
                "Supplier diversity spend: $340M with certified diverse suppliers.",
            ]),
            ("Governance", [
                "Board composition: 75% independent directors.",
                "Executive pay ratio (CEO:Median employee): 42:1.",
                "Whistleblower cases resolved within SLA: 100%.",
            ]),
        ],
    ),
    (
        "Conclusion & Outlook",
        [
            ("Conclusion", [
                "Fiscal year 2025 demonstrated ACME Corporation's resilience and "
                "capacity for innovation. Strong execution across all business units "
                "positioned the company well for continued growth in 2026.",
            ]),
            ("2026 Priorities", [
                "1. Complete cloud migration for remaining 15% of workloads (Q1 2026)",
                "2. Scale AI product suite to 5,000 enterprise accounts",
                "3. Achieve $5B revenue milestone - targeting 19% YoY growth",
                "4. Expand headcount by 2,500 across engineering and sales",
                "5. Reach 90% renewable energy usage globally",
            ]),
            ("Risk Factors", [
                "Macroeconomic uncertainty, competitive pricing pressure in core "
                "markets, and regulatory changes in the EU remain the primary risks "
                "to be monitored closely throughout the coming year.",
            ]),
            (None, [
                "",
                "For questions contact: investor.relations@acme.example.com",
            ]),
        ],
    ),
]

# ── PDF generation ─────────────────────────────────────────────────────────────

def build_pdf(output_path: str) -> None:
    from fpdf import FPDF

    class ReportPDF(FPDF):
        def header(self):
            if self.page_no() == 1:
                return
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(150, 150, 150)
            self.cell(0, 8, "ACME Corporation  |  Annual Performance Report 2025", align="L")
            self.ln(0)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(150, 150, 150)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    pdf = ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(20, 20, 20)

    for page_title, sections in DOCUMENT:
        pdf.add_page()

        # Cover page special layout
        if page_title == "Cover Page":
            pdf.set_y(60)
            pdf.set_font("Helvetica", "B", 28)
            pdf.set_text_color(26, 26, 46)
            for _, lines in sections:
                for line in lines:
                    if line == "ACME CORPORATION":
                        pdf.set_font("Helvetica", "B", 28)
                        pdf.set_text_color(26, 26, 46)
                        pdf.cell(0, 12, line, align="C", new_x="LMARGIN", new_y="NEXT")
                    elif line == "Annual Performance Report":
                        pdf.ln(4)
                        pdf.set_font("Helvetica", "B", 22)
                        pdf.set_text_color(22, 33, 62)
                        pdf.cell(0, 10, line, align="C", new_x="LMARGIN", new_y="NEXT")
                    elif line == "Fiscal Year 2025":
                        pdf.set_font("Helvetica", "", 16)
                        pdf.set_text_color(15, 52, 96)
                        pdf.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")
                    elif line == "CONFIDENTIAL":
                        pdf.ln(4)
                        pdf.set_font("Helvetica", "B", 13)
                        pdf.set_text_color(233, 69, 96)
                        pdf.cell(0, 8, line, align="C", new_x="LMARGIN", new_y="NEXT")
                    elif line.startswith("Prepared") or line.startswith("Date"):
                        pdf.set_font("Helvetica", "", 11)
                        pdf.set_text_color(80, 80, 80)
                        pdf.cell(0, 7, line, align="C", new_x="LMARGIN", new_y="NEXT")
                    else:
                        pdf.ln(max(1, len(line) // 2))
            continue

        # Regular page: page title as H1
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(22, 33, 62)
        pdf.cell(0, 10, page_title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(200, 200, 200)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(4)

        for heading, lines in sections:
            if heading:
                pdf.set_font("Helvetica", "B", 13)
                pdf.set_text_color(15, 52, 96)
                pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)

            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(50, 50, 50)
            for line in lines:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 6, line)
            pdf.ln(4)

    pdf.output(output_path)
    print(f"  wrote {output_path}  ({pdf.page} pages)")


# ── PDF → images ───────────────────────────────────────────────────────────────

def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 150) -> list:
    import fitz  # pymupdf
    from PIL import Image
    import io

    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths = []
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        path = os.path.join(out_dir, f"page_{i:02d}.png")
        img.save(path)
        print(f"  rendered page {i} → {path}")
        paths.append(path)

    doc.close()
    return paths


# ── VLM extraction (via vLLM HTTP server) ─────────────────────────────────────

VLLM_BASE_URL = "http://localhost:8000/v1"

PROMPT_TEXT = (
    "Extract all text visible on this document page. "
    "Preserve headings, bullet points, and layout as closely as possible."
)


async def _process_page(client, model_id: str, i: int, path: str) -> dict:
    """Send one page to the vLLM server and return results."""
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    t_start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model_id,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        }],
        max_tokens=1024,
        temperature=0.1,
    )
    elapsed = time.perf_counter() - t_start
    text       = response.choices[0].message.content.strip()
    tokens_out = response.usage.completion_tokens
    tps        = tokens_out / elapsed if elapsed else 0.0
    return {
        "page":       i,
        "path":       path,
        "text":       text,
        "model_id":   model_id,
        "time_s":     round(elapsed, 3),
        "tokens_out": tokens_out,
        "tps":        round(tps, 1),
    }


async def _extract_parallel(page_paths: list, txt_path: str) -> tuple:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=VLLM_BASE_URL, api_key="dummy")

    print(f"\nConnecting to vLLM server at {VLLM_BASE_URL} …")
    try:
        models = await client.models.list()
        loaded = [m.id for m in models.data]
        if not loaded:
            sys.exit("vLLM server has no models loaded.")
        model_id = loaded[0]
        print(f"  using model: {model_id}")
    except Exception as e:
        sys.exit(
            f"Cannot reach vLLM server: {e}\n"
            "Start it with:\n"
            "  VLLM_WSL2_ENABLE_PIN_MEMORY=1 VLLM_USE_FLASHINFER_SAMPLER=0 \\\n"
            "  vllm serve Qwen/Qwen2.5-VL-3B-Instruct --dtype bfloat16 "
            "--max-model-len 8192 --gpu-memory-utilization 0.85"
        )

    print(f"  sending all {len(page_paths)} pages in parallel …")
    t_wall_start = time.perf_counter()
    results = await asyncio.gather(
        *[_process_page(client, model_id, i, path) for i, path in enumerate(page_paths, 1)]
    )
    total_wall = time.perf_counter() - t_wall_start

    os.makedirs(txt_path.rsplit("/", 1)[0], exist_ok=True)
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
    return total_wall, results


def extract_content(page_paths: list, txt_path: str) -> tuple:
    return asyncio.run(_extract_parallel(page_paths, txt_path))


def print_summary(total_wall: float, page_results: list) -> None:
    n          = len(page_results)
    sum_serial = sum(r["time_s"] for r in page_results)

    print("\n" + "=" * 60)
    print("  vLLM BENCHMARK SUMMARY  (parallel)")
    print("=" * 60)
    print(f"  Backend:              vLLM (HTTP server, parallel)")
    model_id = page_results[0].get("model_id", MODEL) if page_results else MODEL
    print(f"  Model:                {model_id}")
    print(f"  Pages processed:      {n}")
    print(f"  Total wall-clock:     {total_wall:.2f}s  ← actual time taken")
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
    ap = argparse.ArgumentParser(description="PDF content extraction with Qwen2.5-VL-3B via vLLM")
    ap.add_argument(
        "--pdf",
        default=None,
        help="path to an existing PDF; omit to generate the built-in sample",
    )
    ap.add_argument("--dpi", type=int, default=150, help="render resolution (default 150)")
    args = ap.parse_args()

    pdf_path = args.pdf or SAMPLE_PDF

    if args.pdf is None:
        print(f"Generating sample PDF …")
        build_pdf(pdf_path)
    else:
        if not os.path.exists(pdf_path):
            sys.exit(f"error: PDF not found: {pdf_path}")
        print(f"Using existing PDF: {pdf_path}")

    print(f"\nRendering PDF pages to images (dpi={args.dpi}) …")
    page_images = pdf_to_images(pdf_path, PAGE_IMG_DIR, dpi=args.dpi)

    print(f"\nRunning Qwen2.5-VL-3B content extraction on {len(page_images)} pages …")
    total_wall, page_results = extract_content(page_images, OUTPUT_TXT)
    print_summary(total_wall, page_results)


if __name__ == "__main__":
    main()
