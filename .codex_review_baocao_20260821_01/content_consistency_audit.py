from __future__ import annotations

import collections
import json
import re
from pathlib import Path


ROOT = Path(r"E:\SCT_Camera\.codex_review_baocao_20260821_01")
PAGE_TEXT = (ROOT / "page_text.txt").read_text(encoding="utf-8")
STRUCTURE = json.loads((ROOT / "structural_audit.json").read_text(encoding="utf-8"))


parts = re.split(r"===== PAGE (\d+) =====", PAGE_TEXT)
pages = {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}
body_text = "\n".join(pages[i] for i in range(15, 116))
reference_text = "\n".join(pages[i] for i in range(116, 120))


citations = []
for match in re.finditer(r"\[((?:\d+\s*,\s*)*\d+)\]", body_text):
    citations.extend(int(value.strip()) for value in match.group(1).split(","))
reference_numbers = [int(value) for value in re.findall(r"(?m)^\[(\d+)\]", reference_text)]


actual_captions = [
    item
    for item in STRUCTURE["captions"]
    if item["index"] > 100 and item["style"] not in {"toc 1", "table of figures"}
]
caption_numbers = collections.defaultdict(list)
for item in actual_captions:
    match = re.match(r"^(Hình|Bảng)\s+(\d+)-(\d+)\s*:", item["text"])
    if match:
        caption_numbers[match.group(1)].append(
            {
                "chapter": int(match.group(2)),
                "number": int(match.group(3)),
                "text": item["text"],
                "index": item["index"],
                "has_seq_field": any(
                    re.search(r"\bSEQ\b", field, re.IGNORECASE)
                    for field in item["field_instructions"]
                ),
            }
        )


def sequence_issues(items):
    grouped = collections.defaultdict(list)
    for item in items:
        grouped[item["chapter"]].append(item["number"])
    issues = []
    for chapter, numbers in sorted(grouped.items()):
        expected = list(range(1, max(numbers) + 1))
        if numbers != expected:
            issues.append({"chapter": chapter, "found": numbers, "expected": expected})
    return issues


terms = [
    "YOLO11",
    "YOLOv11",
    "YOLO11s",
    "YOLOv11s",
    "YOLO11n-pose",
    "YOLOv11n-pose",
    "YOLOv11-Pose",
    "testcase",
    "test case",
    "test-case",
    "InsightFace",
    "Insightface",
    "ID Switch",
    "ID switch",
]
term_counts = {term: len(re.findall(re.escape(term), body_text)) for term in terms}


output = {
    "citations": {
        "reference_numbers": reference_numbers,
        "missing_reference_numbers": sorted(set(citations) - set(reference_numbers)),
        "uncited_reference_numbers": sorted(set(reference_numbers) - set(citations)),
        "citation_frequency": sorted(collections.Counter(citations).items()),
    },
    "captions": {
        "figure_count": len(caption_numbers["Hình"]),
        "table_count": len(caption_numbers["Bảng"]),
        "figure_sequence_issues": sequence_issues(caption_numbers["Hình"]),
        "table_sequence_issues": sequence_issues(caption_numbers["Bảng"]),
        "manual_numbered_figures": [
            item["text"] for item in caption_numbers["Hình"] if not item["has_seq_field"]
        ],
        "manual_numbered_tables": [
            item["text"] for item in caption_numbers["Bảng"] if not item["has_seq_field"]
        ],
    },
    "term_counts": term_counts,
    "suspicious_reference_lines": [
        line.strip()
        for line in reference_text.splitlines()
        if re.search(r"\b2026\.\s+2016\b|\b2016\.\s+2016\b", line)
    ],
}

print(json.dumps(output, ensure_ascii=False, indent=2))
(ROOT / "content_consistency_audit.json").write_text(
    json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
)
