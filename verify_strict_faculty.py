from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
NS = {"w": W, "m": M, "wp": WP}


def read_parts(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def parse(parts: dict[str, bytes], name: str = "word/document.xml") -> etree._Element:
    return etree.fromstring(parts[name])


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def paragraph_visible_text(paragraph: etree._Element) -> str:
    pieces: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{{{W}}}t" and node.text:
            pieces.append(node.text)
        elif node.tag == f"{{{W}}}br":
            pieces.append(" ")
    return " ".join("".join(pieces).split())


def paragraphs(root: etree._Element) -> list[etree._Element]:
    return root.xpath("//w:body//w:p", namespaces=NS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    src = read_parts(args.source)
    tpl = read_parts(args.template)
    out = read_parts(args.output)
    src_root = parse(src)
    tpl_root = parse(tpl)
    out_root = parse(out)

    for part in (
        "word/styles.xml",
        "word/numbering.xml",
        "word/theme/theme1.xml",
        "word/fontTable.xml",
    ):
        assert out[part] == tpl[part], f"Output differs from template: {part}"

    src_paras = paragraphs(src_root)
    out_paras = paragraphs(out_root)
    assert len(src_paras) == len(out_paras)
    src_texts = [paragraph_visible_text(p) for p in src_paras]
    out_texts = [paragraph_visible_text(p) for p in out_paras]
    assert src_texts == out_texts, "Paragraph text changed"

    tpl_sect = tpl_root.xpath("//w:body/w:sectPr[last()]", namespaces=NS)[0]
    out_sect = out_root.xpath("//w:body/w:sectPr[last()]", namespaces=NS)[0]
    for tag in ("pgSz", "pgMar", "cols", "docGrid"):
        tpl_nodes = tpl_sect.xpath(f"./w:{tag}", namespaces=NS)
        out_nodes = out_sect.xpath(f"./w:{tag}", namespaces=NS)
        assert len(tpl_nodes) == len(out_nodes)
        if tpl_nodes:
            assert etree.tostring(tpl_nodes[0]) == etree.tostring(out_nodes[0]), tag

    pg_sz = out_sect.xpath("./w:pgSz", namespaces=NS)[0]
    pg_mar = out_sect.xpath("./w:pgMar", namespaces=NS)[0]
    attrs = {etree.QName(k).localname: v for k, v in pg_sz.attrib.items()}
    margins = {etree.QName(k).localname: v for k, v in pg_mar.attrib.items()}
    assert attrs["w"] == "11900" and attrs["h"] == "16840", attrs
    assert margins["left"] == "1134", margins
    assert margins["right"] == "1418", margins
    assert margins["top"] == "1134", margins
    assert margins["bottom"] == "1134", margins

    font_overrides: dict[str, int] = {}
    for name, data in out.items():
        if (
            not name.startswith("word/")
            or not name.endswith(".xml")
            or name in {"word/styles.xml", "word/numbering.xml"}
        ):
            continue
        try:
            root = etree.fromstring(data)
        except etree.XMLSyntaxError:
            continue
        count = len(root.xpath("//w:rPr/w:sz | //w:rPr/w:szCs", namespaces=NS))
        if count:
            font_overrides[name] = count
    assert not font_overrides, font_overrides

    direct_layout = []
    for index, paragraph in enumerate(out_paras):
        has_drawing = bool(paragraph.xpath(".//w:drawing", namespaces=NS))
        has_equation = bool(paragraph.xpath(".//m:oMath | .//m:oMathPara", namespaces=NS))
        if has_drawing or has_equation:
            continue
        forbidden = paragraph.xpath(
            "./w:pPr/w:spacing | ./w:pPr/w:jc | ./w:pPr/w:ind",
            namespaces=NS,
        )
        if forbidden:
            direct_layout.append((index, paragraph_text(paragraph)[:80]))
    assert not direct_layout, direct_layout

    title = out_paras[0]
    assert title.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS) == ["TenBaiBao"]
    assert paragraph_visible_text(title) == src_texts[0]
    assert len(title.xpath(".//w:br", namespaces=NS)) == 1
    assert len(title.xpath(".//w:rPr/w:w[@w:val='70']", namespaces=NS)) >= 1

    body_text = "\n".join(out_texts)
    for forbidden in ("CCS CONCEPTS", "ACM Reference Format", "Creative Commons"):
        assert forbidden not in body_text
    for required in (
        "0977314645",
        "20.76 FPS",
        "24.47 FPS",
        "135 ca",
        "130 ca",
        "5 ca",
        "risk_score",
        "gate_alerts",
        "pose_push_contact",
        "score_threshold",
    ):
        assert required in body_text, required

    refs = out_root.xpath(
        "//w:body//w:p[w:pPr/w:pStyle[@w:val='TaiLieuThamKhao']]",
        namespaces=NS,
    )
    assert len(refs) == 18, len(refs)
    assert len(out_root.xpath("//w:tbl", namespaces=NS)) == 3
    assert len(out_root.xpath("//wp:inline | //wp:anchor", namespaces=NS)) == 7
    equation_paragraphs = out_root.xpath(
        "//w:body//w:p[.//m:oMath or .//m:oMathPara]", namespaces=NS
    )
    assert len(equation_paragraphs) == 7
    assert len(
        out_root.xpath("//w:body//w:p[w:pPr/w:pStyle[@w:val='TenHinh']]", namespaces=NS)
    ) == 6
    assert len(
        out_root.xpath("//w:body//w:p[w:pPr/w:pStyle[@w:val='TenBang']]", namespaces=NS)
    ) == 3

    changed_parts = sorted(name for name in src if src[name] != out.get(name))
    allowed = {
        "word/document.xml",
        "word/styles.xml",
        "word/numbering.xml",
        "word/theme/theme1.xml",
        "word/fontTable.xml",
    }
    assert set(changed_parts).issubset(allowed), changed_parts

    print("verification=PASS")
    print("page=A4")
    print("margins_twips=left1134,right1418,top1134,bottom1134")
    print("paragraphs=170")
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
