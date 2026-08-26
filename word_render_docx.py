import os
import sys
from pathlib import Path

import pypdfium2 as pdfium
import pythoncom
import win32com.client


def render(docx_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{docx_path.stem}.pdf"
    pythoncom.CoInitialize()
    word = None
    doc = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        doc = word.Documents.Open(str(docx_path.resolve()), ReadOnly=True, AddToRecentFiles=False)
        doc.Repaginate()
        doc.ExportAsFixedFormat(str(pdf_path.resolve()), 17, OpenAfterExport=False)
        print(f"word_pages={doc.ComputeStatistics(2)}")
    finally:
        if doc is not None:
            doc.Close(False)
        if word is not None:
            word.Quit()
        pythoncom.CoUninitialize()

    pdf = pdfium.PdfDocument(str(pdf_path))
    for i in range(len(pdf)):
        page = pdf[i]
        bitmap = page.render(scale=2.0)
        bitmap.to_pil().save(out_dir / f"page-{i + 1}.png")
    print(f"png_pages={len(pdf)}")


if __name__ == "__main__":
    render(Path(sys.argv[1]), Path(sys.argv[2]))
