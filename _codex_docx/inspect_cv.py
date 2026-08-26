from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def joined_text(element: ET.Element) -> str:
    return "".join(node.text or "" for node in element.findall(".//w:t", NS))


def main() -> None:
    source = Path(sys.argv[1])
    with zipfile.ZipFile(source) as archive:
        doc = ET.fromstring(archive.read("word/document.xml"))
        rels_root = ET.fromstring(archive.read("word/_rels/document.xml.rels"))

        rels = {
            rel.get("Id"): {
                "target": rel.get("Target"),
                "type": rel.get("Type"),
                "mode": rel.get("TargetMode"),
            }
            for rel in rels_root.findall("pr:Relationship", NS)
        }

        blocks = []
        for idx, paragraph in enumerate(doc.findall(".//w:p", NS)):
            text = joined_text(paragraph)
            links = []
            for hyperlink in paragraph.findall(".//w:hyperlink", NS):
                rel_id = hyperlink.get(f"{{{NS['r']}}}id")
                links.append(
                    {
                        "text": joined_text(hyperlink),
                        "rel_id": rel_id,
                        "target": rels.get(rel_id, {}).get("target"),
                    }
                )
            if text or links:
                blocks.append({"index": idx, "text": text, "links": links})

        external_rels = {
            key: value
            for key, value in rels.items()
            if value.get("mode") == "External"
        }

    print(json.dumps({"paragraphs": blocks, "external_relationships": external_rels}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
