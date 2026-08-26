from __future__ import annotations

import sys
from pathlib import Path

import pypdfium2 as pdfium


def main() -> None:
    pdf_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    document = pdfium.PdfDocument(pdf_path)
    try:
        for index in range(len(document)):
            page = document[index]
            bitmap = page.render(scale=2)
            image = bitmap.to_pil()
            output = output_dir / f"page-{index + 1}.png"
            image.save(output)
            print(output)
            bitmap.close()
            page.close()
    finally:
        document.close()


if __name__ == "__main__":
    main()
