from __future__ import annotations

import argparse
import copy
import sys
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"w": W, "m": M, "wp": WP, "a": A}
qn = lambda namespace, tag: f"{{{namespace}}}{tag}"


STYLE_PARTS = (
    "word/styles.xml",
    "word/numbering.xml",
    "word/theme/theme1.xml",
    "word/fontTable.xml",
)


def paragraph_style(paragraph: etree._Element) -> str | None:
    values = paragraph.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
    return values[0] if values else None


def ensure_ppr(paragraph: etree._Element) -> etree._Element:
    ppr = paragraph.find(qn(W, "pPr"))
    if ppr is None:
        ppr = etree.Element(qn(W, "pPr"))
        paragraph.insert(0, ppr)
    return ppr


def ensure_child(parent: etree._Element, namespace: str, tag: str) -> etree._Element:
    child = parent.find(qn(namespace, tag))
    if child is None:
        child = etree.SubElement(parent, qn(namespace, tag))
    return child


def clear_direct_paragraph_format(paragraph: etree._Element, special: str | None) -> None:
    ppr = ensure_ppr(paragraph)
    keep_tags = {qn(W, "pStyle"), qn(W, "numPr"), qn(W, "rPr"), qn(W, "sectPr"), qn(W, "cnfStyle")}
    if special == "equation":
        keep_tags.add(qn(W, "tabs"))
    for child in list(ppr):
        if child.tag not in keep_tags:
            ppr.remove(child)

    if special == "image":
        jc = etree.SubElement(ppr, qn(W, "jc"))
        jc.set(qn(W, "val"), "center")
        spacing = etree.SubElement(ppr, qn(W, "spacing"))
        spacing.set(qn(W, "before"), "0")
        spacing.set(qn(W, "after"), "0")
        ind = etree.SubElement(ppr, qn(W, "ind"))
        ind.set(qn(W, "left"), "0")
        ind.set(qn(W, "right"), "0")
        ind.set(qn(W, "firstLine"), "0")
        keep_next = etree.SubElement(ppr, qn(W, "keepNext"))
        keep_next.set(qn(W, "val"), "1")


def clean_run_font_overrides(root: etree._Element) -> dict[str, int]:
    counts = {"sz": 0, "szCs": 0, "rFonts": 0}
    for tag in ("sz", "szCs"):
        nodes = root.xpath(f".//w:rPr/w:{tag}", namespaces=NS)
        counts[tag] = len(nodes)
        for node in nodes:
            node.getparent().remove(node)

    rfonts = root.xpath(
        ".//w:r[not(ancestor::m:oMath) and not(ancestor::m:oMathPara)]/w:rPr/w:rFonts"
        " | .//w:pPr/w:rPr/w:rFonts",
        namespaces=NS,
    )
    counts["rFonts"] = len(rfonts)
    for node in rfonts:
        node.getparent().remove(node)
    return counts


def set_title_two_lines(root: etree._Element, scale_percent: int) -> str:
    title_paragraphs = root.xpath(
        ".//w:p[w:pPr/w:pStyle[@w:val='TenBaiBao']]", namespaces=NS
    )
    if len(title_paragraphs) != 1:
        raise RuntimeError(f"Expected one title paragraph, found {len(title_paragraphs)}")
    paragraph = title_paragraphs[0]
    old_text = "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))
    expected = "PHÁT TRIỂN HỆ THỐNG CẢNH BÁO THỜI GIAN THỰC DỰA TRÊN AI CHO CAMERA GIÁM SÁT"
    if old_text != expected:
        raise RuntimeError(f"Unexpected title: {old_text!r}")

    for child in list(paragraph):
        if child.tag != qn(W, "pPr"):
            paragraph.remove(child)
    run = etree.SubElement(paragraph, qn(W, "r"))
    rpr = etree.SubElement(run, qn(W, "rPr"))
    width = etree.SubElement(rpr, qn(W, "w"))
    width.set(qn(W, "val"), str(scale_percent))
    first = etree.SubElement(run, qn(W, "t"))
    first.text = "PHÁT TRIỂN HỆ THỐNG CẢNH BÁO THỜI GIAN"
    etree.SubElement(run, qn(W, "br"))
    second = etree.SubElement(run, qn(W, "t"))
    second.text = "THỰC DỰA TRÊN AI CHO CAMERA GIÁM SÁT"
    return old_text


def set_figure_caption_pagination(root: etree._Element) -> tuple[int, int]:
    captions = root.xpath(
        ".//w:p[w:pPr/w:pStyle[@w:val='TenHinh']]", namespaces=NS
    )
    if len(captions) != 6:
        raise RuntimeError(f"Expected 6 figure captions, found {len(captions)}")
    for paragraph in captions:
        ppr = ensure_ppr(paragraph)
        keep_next = ensure_child(ppr, W, "keepNext")
        keep_next.set(qn(W, "val"), "0")

    image_paragraphs = root.xpath(".//w:p[.//w:drawing]", namespaces=NS)
    if len(image_paragraphs) != 6:
        raise RuntimeError(f"Expected 6 image paragraphs, found {len(image_paragraphs)}")
    for paragraph in image_paragraphs:
        ppr = ensure_ppr(paragraph)
        keep_next = ensure_child(ppr, W, "keepNext")
        keep_next.set(qn(W, "val"), "1")
    return len(captions), len(image_paragraphs)


def keep_tables_together(root: etree._Element) -> tuple[int, int]:
    tables = root.xpath("./w:body/w:tbl", namespaces=NS)
    if len(tables) != 3:
        raise RuntimeError(f"Expected 3 body tables, found {len(tables)}")
    cant_split = 0
    keep_next = 0
    for table in tables:
        rows = table.xpath("./w:tr", namespaces=NS)
        for row_index, row in enumerate(rows):
            trpr = row.find(qn(W, "trPr"))
            if trpr is None:
                trpr = etree.Element(qn(W, "trPr"))
                row.insert(0, trpr)
            node = ensure_child(trpr, W, "cantSplit")
            node.set(qn(W, "val"), "1")
            cant_split += 1
            if row_index < len(rows) - 1:
                for paragraph in row.xpath("./w:tc/w:p", namespaces=NS):
                    ppr = ensure_ppr(paragraph)
                    node = ensure_child(ppr, W, "keepNext")
                    node.set(qn(W, "val"), "1")
                    keep_next += 1
    return cant_split, keep_next


def remove_manual_breaks(root: etree._Element) -> int:
    breaks = root.xpath(".//w:br[@w:type='page']", namespaces=NS)
    for node in breaks:
        node.getparent().remove(node)
    return len(breaks)


def apply_template_section(root: etree._Element, template_root: etree._Element) -> None:
    target_sections = root.xpath(".//w:sectPr", namespaces=NS)
    template_sections = template_root.xpath(".//w:sectPr", namespaces=NS)
    if len(target_sections) != 1 or not template_sections:
        raise RuntimeError("Unexpected section structure")
    target = target_sections[0]
    reference = template_sections[-1]
    for tag in ("pgSz", "pgMar", "cols", "docGrid"):
        old = target.find(qn(W, tag))
        new = reference.find(qn(W, tag))
        if old is not None:
            target.remove(old)
        if new is not None:
            target.append(copy.deepcopy(new))


def scale_images(root: etree._Element, scale: float) -> int:
    if scale <= 0 or scale > 1:
        raise ValueError("Image scale must be in (0, 1]")
    extents = root.xpath(".//wp:extent | .//a:xfrm/a:ext", namespaces=NS)
    for extent in extents:
        for attr in ("cx", "cy"):
            if extent.get(attr) is not None:
                extent.set(attr, str(round(int(extent.get(attr)) * scale)))
    return len(root.xpath(".//w:drawing", namespaces=NS))


def patch_document(
    target_xml: bytes,
    template_xml: bytes,
    title_scale: int,
    image_scale: float,
) -> tuple[bytes, dict[str, object]]:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(target_xml, parser)
    template_root = etree.fromstring(template_xml, parser)

    paragraphs = root.xpath(".//w:p", namespaces=NS)
    special_counts = {"image": 0, "equation": 0, "text": 0}
    for paragraph in paragraphs:
        if paragraph.xpath(".//w:drawing", namespaces=NS):
            special = "image"
        elif paragraph.xpath(".//m:oMath | .//m:oMathPara", namespaces=NS):
            special = "equation"
        else:
            special = None
        clear_direct_paragraph_format(paragraph, special)
        special_counts[special or "text"] += 1

    removed_fonts = clean_run_font_overrides(root)
    title_text = set_title_two_lines(root, title_scale)
    captions, image_paragraphs = set_figure_caption_pagination(root)
    cant_split, table_keep_next = keep_tables_together(root)
    manual_breaks = remove_manual_breaks(root)
    apply_template_section(root, template_root)
    drawings = scale_images(root, image_scale)

    remaining_sizes = root.xpath(".//w:rPr/w:sz | .//w:rPr/w:szCs", namespaces=NS)
    if remaining_sizes:
        raise RuntimeError(f"Direct font-size overrides remain: {len(remaining_sizes)}")

    report = {
        "paragraphs": len(paragraphs),
        "paragraph_types": special_counts,
        "removed_font_overrides": removed_fonts,
        "title": title_text,
        "title_scale_percent": title_scale,
        "figure_captions": captions,
        "image_paragraphs": image_paragraphs,
        "drawings": drawings,
        "image_scale": image_scale,
        "table_rows_cant_split": cant_split,
        "table_paragraphs_keep_next": table_keep_next,
        "manual_page_breaks_removed": manual_breaks,
    }
    return (
        etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True),
        report,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--title-scale", type=int, default=80)
    parser.add_argument("--image-scale", type=float, default=1.0)
    args = parser.parse_args()

    source = args.source.resolve()
    template = args.template.resolve()
    output = args.output.resolve()
    if source == output:
        raise RuntimeError("Output must be a new file")

    with zipfile.ZipFile(source, "r") as zin:
        infos = zin.infolist()
        target_parts = {info.filename: zin.read(info.filename) for info in infos}
    with zipfile.ZipFile(template, "r") as ztemplate:
        template_parts = {name: ztemplate.read(name) for name in ztemplate.namelist()}

    patched_xml, report = patch_document(
        target_parts["word/document.xml"],
        template_parts["word/document.xml"],
        args.title_scale,
        args.image_scale,
    )
    replacements = {"word/document.xml": patched_xml}
    for name in STYLE_PARTS:
        replacements[name] = template_parts[name]

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as zout:
        for info in infos:
            data = replacements.get(info.filename, target_parts[info.filename])
            zout.writestr(copy.copy(info), data)

    with zipfile.ZipFile(output, "r") as zcheck:
        output_parts = {name: zcheck.read(name) for name in zcheck.namelist()}
    changed_parts = [
        name for name in target_parts if target_parts[name] != output_parts.get(name)
    ]
    allowed = {"word/document.xml", *STYLE_PARTS}
    if not set(changed_parts).issubset(allowed):
        raise RuntimeError(f"Unexpected changed package parts: {changed_parts}")

    for key, value in report.items():
        print(f"{key}={value}")
    print(f"changed_parts={changed_parts}")
    print(f"output={output}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
