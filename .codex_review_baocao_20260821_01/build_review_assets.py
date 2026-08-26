from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(r"E:\SCT_Camera\.codex_review_baocao_20260821_01")
PDF_PATH = ROOT / "BAOCAO-word.pdf"
PAGES_DIR = ROOT / "pages"
SHEETS_DIR = ROOT / "contact_sheets"


def extract_page_text() -> None:
    pdf = pdfium.PdfDocument(PDF_PATH)
    chunks = []
    for index in range(len(pdf)):
        text_page = pdf[index].get_textpage()
        text = text_page.get_text_range()
        chunks.append(f"\n===== PAGE {index + 1} =====\n{text}")
    (ROOT / "page_text.txt").write_text("".join(chunks), encoding="utf-8")


def build_contact_sheets() -> None:
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    page_paths = sorted(
        PAGES_DIR.glob("page-*.png"),
        key=lambda path: int(path.stem.split("-")[1]),
    )
    thumb_width = 680
    label_height = 34
    gap = 22
    margin = 24
    font = ImageFont.load_default(size=22)

    for sheet_index, start in enumerate(range(0, len(page_paths), 4), start=1):
        selected = page_paths[start : start + 4]
        thumbs = []
        for path in selected:
            with Image.open(path) as source:
                ratio = thumb_width / source.width
                thumb = source.resize(
                    (thumb_width, round(source.height * ratio)), Image.Resampling.LANCZOS
                )
                thumbs.append((path, thumb.copy()))

        thumb_height = max(image.height for _, image in thumbs)
        canvas = Image.new(
            "RGB",
            (
                margin * 2 + thumb_width * 2 + gap,
                margin * 2 + (thumb_height + label_height) * 2 + gap,
            ),
            "#d9d9d9",
        )
        draw = ImageDraw.Draw(canvas)
        for position, (path, thumb) in enumerate(thumbs):
            row, col = divmod(position, 2)
            x = margin + col * (thumb_width + gap)
            y = margin + row * (thumb_height + label_height + gap)
            page_number = int(path.stem.split("-")[1])
            draw.text((x, y), f"PAGE {page_number}", fill="black", font=font)
            canvas.paste(thumb, (x, y + label_height))

        last_page = min(start + 4, len(page_paths))
        canvas.save(
            SHEETS_DIR / f"sheet-{sheet_index:02d}-pages-{start + 1:03d}-{last_page:03d}.jpg",
            quality=90,
        )


if __name__ == "__main__":
    extract_page_text()
    build_contact_sheets()
