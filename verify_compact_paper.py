from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path

from lxml import etree

from compact_paper_content import REPLACEMENTS


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS = {"w": W, "m": M, "wp": WP}


def read_parts(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def parse(parts: dict[str, bytes], name: str = "word/document.xml") -> etree._Element:
    return etree.fromstring(parts[name])


def visible_text(paragraph: etree._Element) -> str:
    pieces: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{{{W}}}t" and node.text:
            pieces.append(node.text)
        elif node.tag == f"{{{W}}}br":
            pieces.append(" ")
    return " ".join("".join(pieces).split())


def paragraph_style(paragraph: etree._Element) -> str | None:
    values = paragraph.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
    return values[0] if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source_parts = read_parts(args.source)
    template_parts = read_parts(args.template)
    output_parts = read_parts(args.output)
    source_root = parse(source_parts)
    template_root = parse(template_parts)
    output_root = parse(output_parts)

    for part in (
        "word/styles.xml",
        "word/numbering.xml",
        "word/theme/theme1.xml",
        "word/fontTable.xml",
    ):
        assert output_parts[part] == template_parts[part], part

    changed_parts = sorted(
        name for name, data in source_parts.items() if data != output_parts.get(name)
    )
    assert changed_parts == ["word/document.xml"], changed_parts

    source_paragraphs = source_root.xpath("//w:body//w:p", namespaces=NS)
    output_paragraphs = output_root.xpath("//w:body//w:p", namespaces=NS)
    assert len(source_paragraphs) == len(output_paragraphs) == 170
    source_texts = [visible_text(p) for p in source_paragraphs]
    output_texts = [visible_text(p) for p in output_paragraphs]

    source_counter = Counter(source_texts)
    output_counter = Counter(output_texts)
    for old, new in REPLACEMENTS.items():
        assert source_counter[old] == 1, old[:80]
        assert output_counter[new] == 1, new[:80]
        source_counter[old] -= 1
        output_counter[new] -= 1
    source_counter += Counter()
    output_counter += Counter()
    assert source_counter == output_counter, "Unexpected text added, removed, or changed"

    source_tables = [
        [visible_text(p) for p in table.xpath(".//w:p", namespaces=NS)]
        for table in source_root.xpath("//w:body/w:tbl", namespaces=NS)
    ]
    output_tables = [
        [visible_text(p) for p in table.xpath(".//w:p", namespaces=NS)]
        for table in output_root.xpath("//w:body/w:tbl", namespaces=NS)
    ]
    assert source_tables == output_tables
    assert len(output_tables) == 3

    source_math = [
        "".join(node.xpath(".//m:t/text()", namespaces=NS))
        for node in source_root.xpath("//w:body//w:p[.//m:oMath or .//m:oMathPara]", namespaces=NS)
    ]
    output_math = [
        "".join(node.xpath(".//m:t/text()", namespaces=NS))
        for node in output_root.xpath("//w:body//w:p[.//m:oMath or .//m:oMathPara]", namespaces=NS)
    ]
    assert source_math == output_math
    assert len(output_math) == 7

    for name, data in source_parts.items():
        if name.startswith("word/media/"):
            assert output_parts[name] == data, name

    template_section = template_root.xpath("//w:body/w:sectPr[last()]", namespaces=NS)[0]
    output_section = output_root.xpath("//w:body/w:sectPr[last()]", namespaces=NS)[0]
    for tag in ("pgSz", "pgMar", "cols", "docGrid"):
        template_node = template_section.xpath(f"./w:{tag}", namespaces=NS)[0]
        output_node = output_section.xpath(f"./w:{tag}", namespaces=NS)[0]
        assert etree.tostring(template_node) == etree.tostring(output_node), tag

    page_size = {
        etree.QName(key).localname: value
        for key, value in output_section.xpath("./w:pgSz", namespaces=NS)[0].attrib.items()
    }
    margins = {
        etree.QName(key).localname: value
        for key, value in output_section.xpath("./w:pgMar", namespaces=NS)[0].attrib.items()
    }
    assert page_size["w"] == "11900" and page_size["h"] == "16840"
    assert margins["left"] == "1134"
    assert margins["right"] == "1418"
    assert margins["top"] == "1134"
    assert margins["bottom"] == "1134"

    direct_sizes: dict[str, int] = {}
    for name, data in output_parts.items():
        if (
            not name.startswith("word/")
            or not name.endswith(".xml")
            or name in {"word/styles.xml", "word/numbering.xml"}
        ):
            continue
        try:
            part_root = etree.fromstring(data)
        except etree.XMLSyntaxError:
            continue
        count = len(part_root.xpath("//w:rPr/w:sz | //w:rPr/w:szCs", namespaces=NS))
        if count:
            direct_sizes[name] = count
    assert not direct_sizes, direct_sizes

    direct_paragraph_layout = []
    for index, paragraph in enumerate(output_paragraphs):
        has_drawing = bool(paragraph.xpath(".//w:drawing", namespaces=NS))
        has_equation = bool(paragraph.xpath(".//m:oMath | .//m:oMathPara", namespaces=NS))
        if has_drawing or has_equation:
            continue
        forbidden = paragraph.xpath(
            "./w:pPr/w:spacing | ./w:pPr/w:jc | ./w:pPr/w:ind", namespaces=NS
        )
        if forbidden:
            direct_paragraph_layout.append(index)
    assert not direct_paragraph_layout, direct_paragraph_layout

    title = output_paragraphs[0]
    assert paragraph_style(title) == "TenBaiBao"
    assert len(title.xpath(".//w:br", namespaces=NS)) == 1
    assert len(title.xpath(".//w:rPr/w:w[@w:val='70']", namespaces=NS)) >= 1

    source_abstract = next(text for text in source_texts if text.startswith("Giám sát video quy mô nhỏ"))
    output_abstract = next(text for text in output_texts if text.startswith("Giám sát video quy mô nhỏ"))
    assert source_abstract == output_abstract
    abstract_words = len(output_abstract.split())
    assert 150 <= abstract_words <= 200, abstract_words
    source_keywords = next(text for text in source_texts if text.startswith("Từ khóa:"))
    output_keywords = next(text for text in output_texts if text.startswith("Từ khóa:"))
    assert source_keywords == output_keywords and output_keywords.count(";") == 3

    references = output_root.xpath(
        "//w:body//w:p[w:pPr/w:pStyle[@w:val='TaiLieuThamKhao']]", namespaces=NS
    )
    source_references = source_root.xpath(
        "//w:body//w:p[w:pPr/w:pStyle[@w:val='TaiLieuThamKhao']]", namespaces=NS
    )
    assert [visible_text(p) for p in references] == [visible_text(p) for p in source_references]
    assert len(references) == 18

    assert len(output_root.xpath("//wp:inline | //wp:anchor", namespaces=NS)) == 7
    assert len(output_root.xpath("//w:body//w:p[w:pPr/w:pStyle[@w:val='TenHinh']]", namespaces=NS)) == 6
    assert len(output_root.xpath("//w:body//w:p[w:pPr/w:pStyle[@w:val='TenBang']]", namespaces=NS)) == 3

    body_text = "\n".join(output_texts)
    for required in (
        "0977314645",
        "20.76 FPS",
        "24.47 FPS",
        "135 ca",
        "130 ca",
        "năm ca chưa đạt",
        "YOLOv11",
        "ByteTrack",
        "InsightFace",
        "hồi quy logistic",
        "risk_score",
        "gate_alerts",
        "pending_person",
        "known_person",
        "pose_push_contact",
        "score_threshold",
        "scripts/train_behavior_classifier.py",
        "/api/behavior-events",
    ):
        assert required in body_text, required
    for forbidden in ("CCS CONCEPTS", "ACM Reference Format", "Creative Commons"):
        assert forbidden not in body_text

    print("verification=PASS")
    print(f"paragraphs_compacted={len(REPLACEMENTS)}")
    print(f"abstract_words={abstract_words}")
    print("page=A4")
    print("margins_cm=left2,right2.5,top2,bottom2")
    print("direct_font_size_overrides=0")
    print("title_lines=2")
    print("tables=3")
    print("figures=6")
    print("drawings=7")
    print("equations=7")
    print("references=18")
    print(f"changed_parts={changed_parts}")


if __name__ == "__main__":
    main()
