import sys
from pathlib import Path

import pypdfium2 as pdfium


pdf_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
pdf = pdfium.PdfDocument(str(pdf_path))
for i in range(len(pdf)):
    page = pdf[i]
    page.render(scale=2.0).to_pil().save(out_dir / f"page-{i + 1}.png")
print(f"png_pages={len(pdf)}")
