from __future__ import annotations

import copy
import hashlib
import sys
import zipfile
from pathlib import Path

import pypdf
from lxml import etree

from edit_paper_three_issues import NS, REFERENCES, W, paragraph_text, qn


def package_parts(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path, "r") as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def pstyle(paragraph: etree._Element) -> str | None:
    styles = paragraph.xpath("./w:pPr/w:pStyle/@w:val", namespaces=NS)
    return styles[0] if styles else None


def remove_added_pagination(
    source_root: etree._Element, output_root: etree._Element
) -> tuple[int, int]:
    source_table = source_root.xpath("./w:body/w:tbl[2]", namespaces=NS)[0]
    output_table = output_root.xpath("./w:body/w:tbl[2]", namespaces=NS)[0]
    source_rows = source_table.xpath("./w:tr", namespaces=NS)
    output_rows = output_table.xpath("./w:tr", namespaces=NS)
    assert len(source_rows) == len(output_rows) == 5

    cant_split = 0
    keep_next = 0
    for source_row, output_row in zip(source_rows, output_rows, strict=True):
        source_cant = source_row.xpath("./w:trPr/w:cantSplit", namespaces=NS)
        output_cant = output_row.xpath("./w:trPr/w:cantSplit", namespaces=NS)
        if not source_cant:
            assert len(output_cant) == 1
            output_cant[0].getparent().remove(output_cant[0])
            cant_split += 1

        source_paragraphs = source_row.xpath("./w:tc/w:p", namespaces=NS)
        output_paragraphs = output_row.xpath("./w:tc/w:p", namespaces=NS)
        assert len(source_paragraphs) == len(output_paragraphs)
        for source_p, output_p in zip(source_paragraphs, output_paragraphs, strict=True):
            source_keep = source_p.xpath("./w:pPr/w:keepNext", namespaces=NS)
            output_keep = output_p.xpath("./w:pPr/w:keepNext", namespaces=NS)
            if not source_keep and output_keep:
                assert len(output_keep) == 1
                output_keep[0].getparent().remove(output_keep[0])
                keep_next += 1

            source_ppr = source_p.find(qn("pPr"))
            output_ppr = output_p.find(qn("pPr"))
            if source_ppr is None and output_ppr is not None and len(output_ppr) == 0:
                output_p.remove(output_ppr)

        source_trpr = source_row.find(qn("trPr"))
        output_trpr = output_row.find(qn("trPr"))
        if source_trpr is None and output_trpr is not None and len(output_trpr) == 0:
            output_row.remove(output_trpr)

    return cant_split, keep_next


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("Usage: verify_three_issues.py SOURCE.docx OUTPUT.docx RENDER.pdf")
    source_path = Path(sys.argv[1]).resolve()
    output_path = Path(sys.argv[2]).resolve()
    pdf_path = Path(sys.argv[3]).resolve()

    source_parts = package_parts(source_path)
    output_parts = package_parts(output_path)
    assert set(source_parts) == set(output_parts)
    changed_parts = [
        name for name in source_parts if source_parts[name] != output_parts[name]
    ]
    assert changed_parts == ["word/document.xml"], changed_parts

    parser = etree.XMLParser(remove_blank_text=False)
    source_root = etree.fromstring(source_parts["word/document.xml"], parser)
    output_root = etree.fromstring(output_parts["word/document.xml"], parser)

    source_paragraphs = source_root.xpath(".//w:p", namespaces=NS)
    output_paragraphs = output_root.xpath(".//w:p", namespaces=NS)
    assert len(source_paragraphs) == len(output_paragraphs)

    text_changes: list[tuple[int, str, str]] = []
    for index, (source_p, output_p) in enumerate(
        zip(source_paragraphs, output_paragraphs, strict=True)
    ):
        source_text = paragraph_text(source_p)
        output_text = paragraph_text(output_p)
        if source_text != output_text:
            text_changes.append((index, source_text, output_text))
            assert pstyle(source_p) == pstyle(output_p)
            assert etree.tostring(source_p.find(qn("pPr")), method="c14n") == etree.tostring(
                output_p.find(qn("pPr")), method="c14n"
            )
            source_runs = source_p.xpath("./w:r/w:rPr", namespaces=NS)
            output_runs = output_p.xpath("./w:r/w:rPr", namespaces=NS)
            assert len(source_runs) == len(output_runs)
            for source_rpr, output_rpr in zip(source_runs, output_runs, strict=True):
                assert etree.tostring(source_rpr, method="c14n") == etree.tostring(
                    output_rpr, method="c14n"
                )

    assert len(text_changes) == 19, len(text_changes)
    phone_changes = [change for change in text_changes if "Điện thoại:" in change[1]]
    assert len(phone_changes) == 1
    assert "[Bổ sung sau]" in phone_changes[0][1]
    assert phone_changes[0][2].endswith("Điện thoại: 0977314645")

    source_refs = [p for p in source_paragraphs if pstyle(p) == "TaiLieuThamKhao"]
    output_refs = [p for p in output_paragraphs if pstyle(p) == "TaiLieuThamKhao"]
    assert len(source_refs) == len(output_refs) == 18
    assert [paragraph_text(p) for p in output_refs] == REFERENCES
    assert all(not paragraph_text(p).startswith("[") for p in output_refs)

    restored_root = copy.deepcopy(output_root)
    restored_paragraphs = restored_root.xpath(".//w:p", namespaces=NS)
    for index, source_text, _ in text_changes:
        text_nodes = restored_paragraphs[index].xpath(".//w:t", namespaces=NS)
        assert len(text_nodes) == 1
        text_nodes[0].text = source_text

    cant_split, keep_next = remove_added_pagination(source_root, restored_root)
    assert cant_split == 5
    assert keep_next == 16
    assert etree.tostring(source_root, method="c14n") == etree.tostring(
        restored_root, method="c14n"
    )

    reader = pypdf.PdfReader(str(pdf_path))
    assert len(reader.pages) == 10
    page_text = [page.extract_text() or "" for page in reader.pages]
    assert "0977314645" in page_text[0]
    assert "[Bổ sung sau]" not in "\n".join(page_text)
    table_2_page = page_text[6]
    for text in (
        "Bảng 2.",
        "Độ trễ theo giai đoạn với một camera.",
        "Phát hiện/Theo dõi",
        "Tư thế",
        "Danh tính",
        "Hành vi",
        "34.81 ms",
        "32.17 ms",
        "0.22 ms",
        "2.45 ms",
    ):
        assert text in table_2_page, text

    print("PASS")
    print("package_parts_changed=word/document.xml_only")
    print("authorized_text_changes=phone_1 references_18")
    print("references_style=TaiLieuThamKhao numbering=automatic")
    print("table_2=5_rows_single_page page=7")
    print("pages=10")
    print(f"output_sha256={hashlib.sha256(output_path.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
