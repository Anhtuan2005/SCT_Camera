from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium


def render_pdf(source: Path, output_dir: Path, scale: float = 1.7) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(source))
    for index in range(len(document)):
        output_path = output_dir / f"page-{index + 1:02d}.png"
        document[index].render(scale=scale).to_pil().save(output_path)
    print(f"png_pages={len(document)}")


if __name__ == "__main__":
    if len(sys.argv) not in {3, 4}:
        raise SystemExit("Usage: render_pdf_pages.py SOURCE.pdf OUTPUT_DIR [SCALE]")
    render_pdf(
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        float(sys.argv[3]) if len(sys.argv) == 4 else 1.7,
    )
