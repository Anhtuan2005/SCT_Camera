# Execution contract - nội dung bổ sung báo cáo

## Reference
- Original retained DOCX: `E:\BAOCAO.docx` (kept unchanged by this task; it is open in Word).
- Read-only snapshot used for evidence: `E:\SCT_Camera\.codex_addendum_baocao_20260821_01\BAOCAO-reference-snapshot.docx`.
- Snapshot SHA-256: `0662abbc86ed8169f17ba0bb95c4119179452738b4e88a581614b66753bea4ef`.
- Reference render: `E:\SCT_Camera\.codex_addendum_baocao_20260821_01\reference.pdf`, 119 pages.
- Prior full-page PNG inspection: `E:\SCT_Camera\.codex_review_baocao_20260821_01\pages`.
- Section evidence: 11 A4 portrait sections.
- Style evidence: `E:\SCT_Camera\.codex_addendum_baocao_20260821_01\style-evidence.json` and `inspect_reference.py` output.

## Page system
- A4 portrait, 8.27 x 11.69 in.
- Margins: left 1.38 in (3.5 cm), right 0.79 in (2.0 cm), top/bottom 1.18 in (3.0 cm).
- One column.
- The standalone insertion pack intentionally omits chapter-specific headers and report page numbers; the user will paste sections into the retained report.

## Typography
- Typeface: Times New Roman for Vietnamese prose and headings.
- Normal: 13 pt, justified, first-line indent 1.0 cm, 1.5 line spacing, 6 pt after.
- Heading 1: 15 pt, bold, centered, 12 pt after, keep with next.
- Heading 2: 14 pt, bold, left, single spacing, 12 pt after, keep with next.
- Heading 3: 14 pt, bold, left, single spacing, 12 pt after, keep with next.
- Caption: 12 pt, italic, centered, 3 pt before, keep together.
- Placeholder text uses the same body font with yellow highlight and the literal marker `[[BỔ SUNG ...]]`.

## Lists and tables
- Real Word bullets/numbering; wrapped lines align under item text.
- Tables use explicit widths within 15.5 cm usable width, fixed column grids, 0.12 cm cell margins, vertically centered cells, and repeating bold header rows.
- Table body is 11.5-12 pt when density requires it; this is the only deliberate type-size exception.
- No fixed row heights. Rows may expand and split only when necessary.
- Table captions are above tables, centered and italic.

## Content flow and slot map
- Opening note: explains that this file contains only insertable additions and that placeholders require real measurements.
- Chapter 4 block: ablation study, qualitative error analysis, statistical reliability/reproducibility.
- Chapter 5 block: privacy/security/ethics and threats to validity.
- Appendix block: experiment record and error-case record templates.
- The `4.x`/`5.x` numbering is intentionally unresolved because the retained report must assign final numbering after insertion.
- All result cells marked `[[BỔ SUNG ...]]` must be replaced by measured data; no value may be invented.

## Package preservation and fidelity gates
- Reference content, images, relationships, comments, fields, headers, and footers are preserve-only and are not copied into the final standalone file.
- Reuse only the page geometry, typography rhythm, heading hierarchy, caption treatment, and restrained table treatment.
- Final DOCX must contain no source-report body content outside the new sections.
- Render every final page; fail on clipping, overlap, broken tables, isolated headings, or unexplained blank pages.
