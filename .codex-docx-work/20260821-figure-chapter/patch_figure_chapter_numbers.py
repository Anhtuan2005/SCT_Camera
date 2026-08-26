from __future__ import annotations

import argparse
import copy
import re
import zipfile
from pathlib import Path

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}
W = f"{{{W_NS}}}"


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS)).strip()


def result_run(template: etree._Element | None, text: str) -> etree._Element:
    run = copy.deepcopy(template) if template is not None else etree.Element(W + "r")
    for child in list(run):
        if child.tag != W + "rPr":
            run.remove(child)
    node = etree.SubElement(run, W + "t")
    node.text = text
    return run


def set_field_result(field: etree._Element, text: str) -> None:
    text_nodes = field.xpath(".//w:t", namespaces=NS)
    if text_nodes:
        text_nodes[0].text = text
        for node in text_nodes[1:]:
            node.text = ""
        return
    field.append(result_run(None, text))


def patch_document(xml_bytes: bytes) -> tuple[bytes, int]:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml_bytes, parser)
    chapter = 0
    per_chapter: dict[int, int] = {}
    changed = 0

    for paragraph in root.xpath("//w:body//w:p", namespaces=NS):
        style = paragraph.xpath("string(./w:pPr/w:pStyle/@w:val)", namespaces=NS)
        text = paragraph_text(paragraph).upper()
        if style == "Heading1" and text not in {"LỜI CAM ĐOAN", "TÀI LIỆU THAM KHẢO"}:
            chapter += 1

        for field in paragraph.xpath(".//w:fldSimple", namespaces=NS):
            instruction = field.get(W + "instr", "")
            if not re.search(r"\bSEQ\s+Hình\b", instruction, flags=re.IGNORECASE):
                continue
            if chapter < 1:
                raise RuntimeError(f"Figure caption found before Chapter 1: {paragraph_text(paragraph)!r}")

            per_chapter[chapter] = per_chapter.get(chapter, 0) + 1
            sequence = per_chapter[chapter]
            template_run = field.find(W + "r")
            parent = field.getparent()
            index = parent.index(field)

            chapter_field = etree.Element(W + "fldSimple")
            chapter_field.set(W + "instr", ' STYLEREF "Heading 1" \\s ')
            chapter_field.append(result_run(template_run, str(chapter)))
            parent.insert(index, chapter_field)
            parent.insert(index + 1, result_run(template_run, "-"))

            field.set(W + "instr", " SEQ Hình \\* ARABIC \\s 1 ")
            set_field_result(field, str(sequence))
            changed += 1

    if changed != 37:
        raise RuntimeError(f"Expected 37 figure captions, found {changed}")

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes"), changed


def patch_settings(xml_bytes: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml_bytes, parser)
    update = root.find(W + "updateFields")
    if update is None:
        update = etree.SubElement(root, W + "updateFields")
    update.set(W + "val", "true")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Input and output must be different files")

    with zipfile.ZipFile(args.input, "r") as source:
        document_xml, changed = patch_document(source.read("word/document.xml"))
        settings_xml = patch_settings(source.read("word/settings.xml"))
        with zipfile.ZipFile(args.output, "w") as target:
            for item in source.infolist():
                data = source.read(item.filename)
                if item.filename == "word/document.xml":
                    data = document_xml
                elif item.filename == "word/settings.xml":
                    data = settings_xml
                target.writestr(item, data)

    print(f"Updated {changed} figure captions in {args.output}")


if __name__ == "__main__":
    main()
