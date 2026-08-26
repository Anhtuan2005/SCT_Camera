from __future__ import annotations

import sys
from pathlib import Path

import pythoncom
import win32com.client


def export_docx(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    pythoncom.CoInitialize()
    word = None
    document = None
    try:
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        word.DisplayAlerts = 0
        document = word.Documents.Open(
            str(source.resolve()), ReadOnly=True, AddToRecentFiles=False
        )
        document.Repaginate()
        print(f"word_pages={document.ComputeStatistics(2)}")
        document.ExportAsFixedFormat(
            str(destination.resolve()), 17, OpenAfterExport=False
        )
    finally:
        if document is not None:
            document.Close(False)
        if word is not None:
            word.Quit()
        pythoncom.CoUninitialize()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: export_docx_word_pdf.py SOURCE.docx OUTPUT.pdf")
    export_docx(Path(sys.argv[1]), Path(sys.argv[2]))
