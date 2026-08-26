from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: render_pdf_pages.py INPUT.pdf OUTPUT_DIR [DPI]")
    source = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    dpi = float(sys.argv[3]) if len(sys.argv) == 4 else 150.0
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdfium.PdfDocument(source)
    scale = dpi / 72.0
    for index in range(len(pdf)):
        page = pdf[index]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        image.save(output_dir / f"page-{index + 1}.png")
        page.close()
    print(f"PAGES\t{len(pdf)}")
    print(f"OUTPUT_DIR\t{output_dir}")


if __name__ == "__main__":
    main()
