from __future__ import annotations

import copy
import sys
import zipfile
from pathlib import Path

from lxml import etree


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: remove_direct_font_sizes.py SOURCE.docx OUTPUT.docx")
    source = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    if source == output:
        raise RuntimeError("Output must be a new file")

    with zipfile.ZipFile(source, "r") as zin:
        infos = zin.infolist()
        parts = {info.filename: zin.read(info.filename) for info in infos}

    changed_parts: list[str] = []
    removed_by_part: dict[str, int] = {}
    patched_parts = dict(parts)
    parser = etree.XMLParser(remove_blank_text=False)

    for name, data in parts.items():
        if not name.startswith("word/") or not name.endswith(".xml"):
            continue
        if name in {"word/styles.xml", "word/numbering.xml"}:
            continue
        root = etree.fromstring(data, parser)
        nodes = root.xpath(".//w:rPr/w:sz | .//w:rPr/w:szCs", namespaces=NS)
        if not nodes:
            continue
        for node in nodes:
            node.getparent().remove(node)
        patched_parts[name] = etree.tostring(
            root, xml_declaration=True, encoding="UTF-8", standalone=True
        )
        changed_parts.append(name)
        removed_by_part[name] = len(nodes)

    if sum(removed_by_part.values()) == 0:
        raise RuntimeError("No direct font-size overrides were found")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as zout:
        for info in infos:
            zout.writestr(copy.copy(info), patched_parts[info.filename])

    with zipfile.ZipFile(output, "r") as zcheck:
        output_parts = {name: zcheck.read(name) for name in zcheck.namelist()}
    actual_changed = [name for name in parts if parts[name] != output_parts.get(name)]
    if actual_changed != changed_parts:
        raise RuntimeError(
            f"Unexpected package changes: expected {changed_parts}, got {actual_changed}"
        )

    remaining = 0
    for name, data in output_parts.items():
        if not name.startswith("word/") or not name.endswith(".xml"):
            continue
        if name in {"word/styles.xml", "word/numbering.xml"}:
            continue
        root = etree.fromstring(data, parser)
        remaining += len(
            root.xpath(".//w:rPr/w:sz | .//w:rPr/w:szCs", namespaces=NS)
        )
    if remaining:
        raise RuntimeError(f"Direct font-size overrides remain: {remaining}")

    print(f"removed={sum(removed_by_part.values())}")
    print(f"removed_by_part={removed_by_part}")
    print(f"changed_parts={actual_changed}")
    print(f"output={output}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
