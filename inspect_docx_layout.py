from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


path = Path(sys.argv[1])
with zipfile.ZipFile(path) as archive:
    root = etree.fromstring(archive.read("word/document.xml"))
for table_index, table in enumerate(root.xpath("//w:body/w:tbl", namespaces=NS), 1):
    print(f"table={table_index}")
    tbl_pr = table.xpath("./w:tblPr", namespaces=NS)[0]
    for node in tbl_pr:
        if etree.QName(node).localname in {"tblW", "tblInd", "tblCellMar", "tblLayout"}:
            print(etree.tostring(node, encoding="unicode"))
    for row_index, row in enumerate(table.xpath("./w:tr", namespaces=NS), 1):
        tr_pr = row.xpath("./w:trPr", namespaces=NS)
        heights = row.xpath("./w:trPr/w:trHeight", namespaces=NS)
        print(
            f" row={row_index} trPr={bool(tr_pr)} heights="
            f"{[dict(node.attrib) for node in heights]}"
        )
    margins = table.xpath(".//w:tcPr/w:tcMar", namespaces=NS)
    print(f" cell_margin_overrides={len(margins)}")
    unique_margins = sorted(
        {
            tuple(
                (etree.QName(child).localname, tuple(sorted(child.attrib.items())))
                for child in margin
            )
            for margin in margins
        },
        key=repr,
    )
    print(f" unique_cell_margins={unique_margins}")
