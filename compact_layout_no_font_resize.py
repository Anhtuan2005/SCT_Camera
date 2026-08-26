from __future__ import annotations

import argparse
import copy
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}
qn = lambda tag: f"{{{W}}}{tag}"


def ensure(parent: etree._Element, tag: str, first: bool = False) -> etree._Element:
    child = parent.find(qn(tag))
    if child is None:
        child = etree.Element(qn(tag))
        if first:
            parent.insert(0, child)
        else:
            parent.append(child)
    return child


def pstyle(paragraph: etree._Element) -> str | None:
    values = paragraph.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
    return values[0] if values else None


def patch_document(
    xml: bytes,
    horizontal_margin_cm: float | None,
    vertical_margin_cm: float | None,
) -> tuple[bytes, dict[str, int | float | None]]:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml, parser)

    captions = 0
    references = 0
    for paragraph in root.xpath(".//w:p", namespaces=NS):
        style = pstyle(paragraph)
        if style == "TenHinh":
            p_pr = ensure(paragraph, "pPr", first=True)
            keep_next = ensure(p_pr, "keepNext")
            keep_next.set(qn("val"), "0")
            captions += 1
        elif style == "TaiLieuThamKhao":
            p_pr = ensure(paragraph, "pPr", first=True)
            spacing = ensure(p_pr, "spacing")
            spacing.set(qn("before"), "0")
            spacing.set(qn("after"), "0")
            spacing.set(qn("line"), "280")
            spacing.set(qn("lineRule"), "atLeast")
            references += 1

    if captions != 6:
        raise RuntimeError(f"Expected 6 figure captions, found {captions}")
    if references != 18:
        raise RuntimeError(f"Expected 18 references, found {references}")

    sect_prs = root.xpath(".//w:sectPr", namespaces=NS)
    if len(sect_prs) != 1:
        raise RuntimeError(f"Expected one section, found {len(sect_prs)}")
    pg_mar = ensure(sect_prs[0], "pgMar")
    if horizontal_margin_cm is not None:
        horizontal_twips = round(horizontal_margin_cm / 2.54 * 1440)
        pg_mar.set(qn("left"), str(horizontal_twips))
        pg_mar.set(qn("right"), str(horizontal_twips))
    if vertical_margin_cm is not None:
        vertical_twips = round(vertical_margin_cm / 2.54 * 1440)
        pg_mar.set(qn("top"), str(vertical_twips))
        pg_mar.set(qn("bottom"), str(vertical_twips))

    remaining_sizes = root.xpath(".//w:rPr/w:sz | .//w:rPr/w:szCs", namespaces=NS)
    if remaining_sizes:
        raise RuntimeError(f"Direct font-size overrides remain: {len(remaining_sizes)}")

    return (
        etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True),
        {
            "figure_captions_unlinked": captions,
            "references_compacted": references,
            "horizontal_margin_cm": horizontal_margin_cm,
            "vertical_margin_cm": vertical_margin_cm,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--horizontal-margin-cm", type=float)
    parser.add_argument("--vertical-margin-cm", type=float)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise RuntimeError("Output must be a new file")

    with zipfile.ZipFile(source, "r") as zin:
        infos = zin.infolist()
        parts = {info.filename: zin.read(info.filename) for info in infos}

    patched_xml, report = patch_document(
        parts["word/document.xml"],
        args.horizontal_margin_cm,
        args.vertical_margin_cm,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as zout:
        for info in infos:
            data = patched_xml if info.filename == "word/document.xml" else parts[info.filename]
            zout.writestr(copy.copy(info), data)

    with zipfile.ZipFile(output, "r") as zcheck:
        output_parts = {name: zcheck.read(name) for name in zcheck.namelist()}
    changed_parts = [name for name in parts if parts[name] != output_parts.get(name)]
    if changed_parts != ["word/document.xml"]:
        raise RuntimeError(f"Unexpected changed package parts: {changed_parts}")

    for key, value in report.items():
        print(f"{key}={value}")
    print(f"changed_parts={changed_parts}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
