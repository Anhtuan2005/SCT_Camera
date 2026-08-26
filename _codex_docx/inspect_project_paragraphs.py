from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


def text_of(element) -> str:
    return "".join(node.text or "" for node in element.iter(qn("w:t")))


document = Document(Path(sys.argv[1]))
for paragraph_index, paragraph in enumerate(document.element.body.iterchildren(qn("w:p"))):
    paragraph_text = text_of(paragraph)
    if not paragraph_text.startswith("STACK"):
        continue
    print(f"PARAGRAPH {paragraph_index}: {paragraph_text}")
    for child_index, child in enumerate(paragraph):
        tag = child.tag.rsplit("}", 1)[-1]
        rel_id = child.get(qn("r:id"), "")
        print(f"  {child_index}: tag={tag} rel={rel_id!r} text={text_of(child)!r}")
        if tag == "hyperlink":
            print(child.xml)
