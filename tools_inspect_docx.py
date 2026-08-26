import hashlib
import glob
import json
import sys
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


def length(v):
    return None if v is None else int(v)


def style_info(style):
    pf = style.paragraph_format
    font = style.font
    return {
        "name": style.name,
        "id": style.style_id,
        "type": int(style.type),
        "base": style.base_style.name if style.base_style else None,
        "font": font.name,
        "size_pt": font.size.pt if font.size else None,
        "bold": font.bold,
        "italic": font.italic,
        "all_caps": font.all_caps,
        "color": str(font.color.rgb) if font.color and font.color.rgb else None,
        "align": int(pf.alignment) if pf.alignment is not None else None,
        "before_twips": length(pf.space_before),
        "after_twips": length(pf.space_after),
        "line_spacing": str(pf.line_spacing),
        "line_rule": int(pf.line_spacing_rule) if pf.line_spacing_rule is not None else None,
        "left_indent": length(pf.left_indent),
        "right_indent": length(pf.right_indent),
        "first_line_indent": length(pf.first_line_indent),
        "keep_with_next": pf.keep_with_next,
        "keep_together": pf.keep_together,
        "page_break_before": pf.page_break_before,
    }


def run_summary(run):
    rpr = run._r.rPr
    return {
        "text": run.text,
        "bold": run.bold,
        "italic": run.italic,
        "font": run.font.name,
        "size_pt": run.font.size.pt if run.font.size else None,
        "has_drawing": bool(run._r.xpath('.//w:drawing')),
        "has_omath": bool(run._r.xpath('.//m:oMath | .//m:oMathPara')),
        "rfonts": rpr.rFonts.attrib if rpr is not None and rpr.rFonts is not None else None,
    }


def inspect(path):
    path = Path(path)
    doc = Document(str(path))
    required = [
        "TenBaiBao", "TacGia", "TomTat", "TuKhoa", "MucChinh",
        "MucPhu1", "MucPhu2", "NoiDung", "TenBang", "TenHinh",
        "TieuDeBang", "NoiDungBang", "TaiLieuThamKhao",
    ]
    sections = []
    for i, s in enumerate(doc.sections):
        sect = s._sectPr
        cols = sect.find(qn("w:cols"))
        sections.append({
            "index": i,
            "width_twips": length(s.page_width),
            "height_twips": length(s.page_height),
            "orientation": int(s.orientation),
            "margin_top": length(s.top_margin),
            "margin_bottom": length(s.bottom_margin),
            "margin_left": length(s.left_margin),
            "margin_right": length(s.right_margin),
            "header_distance": length(s.header_distance),
            "footer_distance": length(s.footer_distance),
            "start_type": int(s.start_type),
            "cols_num": cols.get(qn("w:num")) if cols is not None else None,
            "cols_space": cols.get(qn("w:space")) if cols is not None else None,
            "different_first": s.different_first_page_header_footer,
            "header_linked": s.header.is_linked_to_previous,
            "footer_linked": s.footer.is_linked_to_previous,
        })

    paras = []
    for i, p in enumerate(doc.paragraphs):
        paras.append({
            "i": i,
            "style": p.style.name if p.style else None,
            "text": p.text,
            "align": int(p.alignment) if p.alignment is not None else None,
            "runs": [run_summary(r) for r in p.runs],
            "has_omath": bool(p._p.xpath('.//m:oMath | .//m:oMathPara')),
            "has_drawing": bool(p._p.xpath('.//w:drawing')),
            "sectPr": bool(p._p.xpath('./w:pPr/w:sectPr')),
        })

    tables = []
    for ti, t in enumerate(doc.tables):
        rows = []
        for ri, row in enumerate(t.rows):
            cells = []
            for ci, cell in enumerate(row.cells):
                cells.append({
                    "text": cell.text,
                    "styles": [p.style.name if p.style else None for p in cell.paragraphs],
                    "paragraphs": [p.text for p in cell.paragraphs],
                })
            rows.append(cells)
        grid = t._tbl.tblGrid
        tables.append({
            "i": ti,
            "style": t.style.name if t.style else None,
            "rows": len(t.rows),
            "cols": len(t.columns),
            "grid": [int(gc.w) for gc in grid.gridCol_lst] if grid is not None else [],
            "data": rows,
        })

    body = []
    p_seq = 0
    t_seq = 0
    for child in doc.element.body.iterchildren():
        tag = child.tag.rsplit('}', 1)[-1]
        if tag == "p":
            body.append({"kind": "p", "index": p_seq})
            p_seq += 1
        elif tag == "tbl":
            body.append({"kind": "tbl", "index": t_seq})
            t_seq += 1
        elif tag == "sectPr":
            body.append({"kind": "sectPr"})
        else:
            body.append({"kind": tag})

    images = []
    for i, shape in enumerate(doc.inline_shapes):
        images.append({"i": i, "width_emu": int(shape.width), "height_emu": int(shape.height), "type": str(shape.type)})

    headers = []
    footers = []
    for i, s in enumerate(doc.sections):
        headers.append({"section": i, "paragraphs": [(p.style.name if p.style else None, p.text) for p in s.header.paragraphs]})
        footers.append({"section": i, "paragraphs": [(p.style.name if p.style else None, p.text) for p in s.footer.paragraphs]})

    with zipfile.ZipFile(path) as zf:
        parts = [{"name": z.filename, "size": z.file_size, "sha256": hashlib.sha256(zf.read(z.filename)).hexdigest()} for z in zf.infolist()]

    available = {s.name: s for s in doc.styles}
    styles = {name: style_info(available[name]) if name in available else None for name in required}
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
        "paragraph_count": len(paras),
        "table_count": len(tables),
        "image_count": len(images),
        "section_count": len(sections),
        "sections": sections,
        "styles": styles,
        "paragraphs": paras,
        "tables": tables,
        "body": body,
        "images": images,
        "headers": headers,
        "footers": footers,
        "parts": parts,
    }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    arg = sys.argv[1]
    if any(ch in arg for ch in "*?"):
        matches = glob.glob(arg)
        if len(matches) != 1:
            raise SystemExit(f"Expected one match for {arg!r}, got {matches}")
        arg = matches[0]
    result = inspect(arg)
    mode = sys.argv[2] if len(sys.argv) > 2 else None
    if mode == "concise":
        result = {
            "path": result["path"], "sha256": result["sha256"], "size": result["size"],
            "paragraph_count": result["paragraph_count"], "table_count": result["table_count"],
            "image_count": result["image_count"], "section_count": result["section_count"],
            "sections": result["sections"], "styles": result["styles"],
            "paragraphs": [
                {k: p[k] for k in ("i", "style", "text", "align", "has_omath", "has_drawing", "sectPr")}
                for p in result["paragraphs"]
            ],
            "tables": result["tables"], "images": result["images"],
            "headers": result["headers"], "footers": result["footers"], "body": result["body"],
        }
    elif mode == "paras":
        start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        end = int(sys.argv[4]) if len(sys.argv) > 4 else result["paragraph_count"]
        result = {
            "path": result["path"],
            "paragraphs": [
                {k: p[k] for k in ("i", "style", "text", "align", "has_omath", "has_drawing", "sectPr")}
                for p in result["paragraphs"][start:end]
            ],
            "body": [x for x in result["body"] if x.get("kind") != "p" or start <= x.get("index", -1) < end],
        }
    elif mode == "stats":
        result = {
            "path": result["path"],
            "paragraph_words": sum(len(p["text"].split()) for p in result["paragraphs"]),
            "table_words": sum(
                len(cell["text"].split())
                for table in result["tables"]
                for row in table["data"]
                for cell in row
            ),
            "paragraph_count": result["paragraph_count"],
            "table_count": result["table_count"],
            "image_count": result["image_count"],
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
