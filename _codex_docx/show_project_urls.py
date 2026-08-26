from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


def main() -> None:
    source = Path(sys.argv[1])
    output = Path(sys.argv[2])
    document = Document(source)
    changed = []

    for hyperlink in document.element.iter(qn("w:hyperlink")):
        rel_id = hyperlink.get(qn("r:id"))
        if not rel_id or rel_id not in document.part.rels:
            continue

        target = document.part.rels[rel_id].target_ref
        text_nodes = list(hyperlink.iter(qn("w:t")))
        visible_text = "".join(node.text or "" for node in text_nodes)
        if visible_text.strip() != "Repository" or not target.startswith("https://github.com/"):
            continue

        text_nodes[0].text = target
        for node in text_nodes[1:]:
            node.text = ""
        changed.append((visible_text, target))

    if len(changed) != 2:
        raise RuntimeError(f"Expected 2 project links, changed {len(changed)}: {changed}")

    output.parent.mkdir(parents=True, exist_ok=True)
    document.save(output)

    check = Document(output)
    verified = []
    for hyperlink in check.element.iter(qn("w:hyperlink")):
        rel_id = hyperlink.get(qn("r:id"))
        if not rel_id or rel_id not in check.part.rels:
            continue
        target = check.part.rels[rel_id].target_ref
        visible_text = "".join(node.text or "" for node in hyperlink.iter(qn("w:t")))
        if target in {item[1] for item in changed}:
            verified.append((visible_text, target))

    if len(verified) != 2 or any(text != target for text, target in verified):
        raise RuntimeError(f"Hyperlink verification failed: {verified}")

    for text, target in verified:
        print(f"OK: {text} -> {target}")


if __name__ == "__main__":
    main()
