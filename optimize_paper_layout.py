from __future__ import annotations

import argparse
import copy
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
qn = lambda tag: f"{{{W}}}{tag}"


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def style_id(paragraph: etree._Element) -> str | None:
    values = paragraph.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
    return values[0] if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with zipfile.ZipFile(args.input, "r") as source:
        infos = source.infolist()
        parts = {info.filename: source.read(info.filename) for info in infos}

    root = etree.fromstring(parts["word/document.xml"])
    body = root.find(qn("body"))
    assert body is not None
    children = list(body)

    caption = next(
        node
        for node in children
        if node.tag == qn("p")
        and style_id(node) == "TenHinh"
        and "AlertManager" in paragraph_text(node)
    )
    caption_index = children.index(caption)
    image_paragraph = children[caption_index - 1]
    assert image_paragraph.xpath(".//w:drawing", namespaces=NS)

    body_paragraph = next(
        node
        for node in children[caption_index + 1 :]
        if node.tag == qn("p") and style_id(node) == "NoiDung"
    )
    assert "asyncio.Queue" in paragraph_text(body_paragraph)

    body.remove(image_paragraph)
    body.remove(caption)
    insert_at = list(body).index(body_paragraph) + 1
    body.insert(insert_at, image_paragraph)
    body.insert(insert_at + 1, caption)

    replacements = {
        "word/document.xml": etree.tostring(
            root, xml_declaration=True, encoding="UTF-8", standalone=True
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w") as target:
        for info in infos:
            target.writestr(copy.copy(info), replacements.get(info.filename, parts[info.filename]))

    print("moved=Hinh 4 after its explanatory paragraph")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
