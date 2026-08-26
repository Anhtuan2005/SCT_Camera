from __future__ import annotations

import sys

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn


def main():
    document = Document(sys.argv[1])
    for name in ("Normal", "List Paragraph", "Default Paragraph Font", "Heading 1", "Heading 2", "Heading 3"):
        style = document.styles[name]
        chain = []
        current = style
        while current is not None:
            chain.append(
                {
                    "name": current.name,
                    "type": str(current.type),
                    "size": current.font.size.pt if current.font.size else None,
                    "font": current.font.name,
                    "bold": current.font.bold,
                }
            )
            current = current.base_style
        print(name, chain, sep="\t")

    styles_root = document.styles.element
    doc_defaults = styles_root.find("./w:docDefaults/w:rPrDefault/w:rPr", styles_root.nsmap)
    if doc_defaults is not None:
        size = doc_defaults.find("./w:sz", styles_root.nsmap)
        fonts = doc_defaults.find("./w:rFonts", styles_root.nsmap)
        print(
            "docDefaults",
            {
                "size_half_points": size.get(qn("w:val")) if size is not None else None,
                "ascii": fonts.get(qn("w:ascii")) if fonts is not None else None,
                "hAnsi": fonts.get(qn("w:hAnsi")) if fonts is not None else None,
            },
            sep="\t",
        )


if __name__ == "__main__":
    main()
