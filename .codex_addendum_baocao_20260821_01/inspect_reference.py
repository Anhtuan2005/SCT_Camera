from __future__ import annotations

import json
from pathlib import Path

from docx import Document


SOURCE = Path(r"E:\SCT_Camera\.codex_addendum_baocao_20260821_01\BAOCAO-reference-snapshot.docx")
doc = Document(SOURCE)


def length(value):
    return None if value is None else {"pt": value.pt, "cm": value.cm}


def style_info(name: str):
    style = doc.styles[name]
    font = style.font
    pf = style.paragraph_format
    return {
        "name": name,
        "font": font.name,
        "size_pt": None if font.size is None else font.size.pt,
        "bold": font.bold,
        "italic": font.italic,
        "alignment": None if pf.alignment is None else int(pf.alignment),
        "space_before": length(pf.space_before),
        "space_after": length(pf.space_after),
        "line_spacing": pf.line_spacing,
        "first_line_indent": length(pf.first_line_indent),
        "left_indent": length(pf.left_indent),
        "keep_with_next": pf.keep_with_next,
        "keep_together": pf.keep_together,
    }


styles = {}
for name in ["Normal", "Heading 1", "Heading 2", "Heading 3", "Caption", "List Paragraph"]:
    if name in doc.styles:
        styles[name] = style_info(name)

samples = []
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    if p.style.name.lower() not in {"toc 1", "table of figures"} and text.startswith(
        ("CHƯƠNG 4", "4.1 ", "4.5.1 ", "Bảng 4-", "Hình 4-")
    ):
        samples.append(
            {
                "index": i,
                "style": p.style.name,
                "text": text[:160],
                "runs": [
                    {
                        "text": run.text[:80],
                        "font": run.font.name,
                        "size_pt": None if run.font.size is None else run.font.size.pt,
                        "bold": run.bold,
                        "italic": run.italic,
                    }
                    for run in p.runs[:4]
                ],
            }
        )
    if len(samples) >= 12:
        break

print(json.dumps({"styles": styles, "samples": samples}, ensure_ascii=False, indent=2))
