from __future__ import annotations

import re
from pathlib import Path


SOURCE = Path(r"E:\SCT_Camera\.codex_review_baocao_20260821_01\page_text.txt")
text = SOURCE.read_text(encoding="utf-8")
parts = re.split(r"===== PAGE (\d+) =====", text)
pages = {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}
pattern = re.compile(r"\[(?:\d+\s*,\s*)*\d+\]")

for page_number in range(15, 116):
    lines = [line.strip() for line in pages[page_number].splitlines()]
    hits = []
    for index, line in enumerate(lines):
        if not pattern.search(line):
            continue
        context = " ".join(lines[max(0, index - 1) : min(len(lines), index + 2)])
        context = re.sub(r"\s+", " ", context).strip()
        hits.append(context)
    if hits:
        print(f"PHYSICAL_PAGE={page_number} PRINTED_PAGE={page_number - 14}")
        for hit in hits:
            print(f"  {hit}")
