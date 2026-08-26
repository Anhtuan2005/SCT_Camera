from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


PDF = Path(r"E:\SCT_Camera\.codex_addendum_baocao_20260821_01\qa-03\NOI_DUNG_BO_SUNG_BAO_CAO.pdf")
OUT = PDF.parent / "pages"
CONTACTS = PDF.parent / "contact-sheets"


def render_pages():
    OUT.mkdir(parents=True, exist_ok=True)
    document = pdfium.PdfDocument(str(PDF))
    paths = []
    for index in range(len(document)):
        page = document[index]
        bitmap = page.render(scale=2.0)
        image = bitmap.to_pil().convert("RGB")
        path = OUT / f"page-{index + 1:03d}.png"
        image.save(path, "PNG")
        paths.append(path)
    return paths


def make_contacts(paths):
    CONTACTS.mkdir(parents=True, exist_ok=True)
    thumb_width = 700
    gutter = 30
    label_height = 45
    for sheet_index, start in enumerate(range(0, len(paths), 4), start=1):
        batch = paths[start : start + 4]
        thumbs = []
        for path in batch:
            image = Image.open(path).convert("RGB")
            height = round(image.height * thumb_width / image.width)
            thumbs.append(image.resize((thumb_width, height)))
        cell_height = max(image.height for image in thumbs) + label_height
        sheet = Image.new("RGB", (thumb_width * 2 + gutter * 3, cell_height * 2 + gutter * 3), "#d8d8d8")
        draw = ImageDraw.Draw(sheet)
        for offset, image in enumerate(thumbs):
            row, col = divmod(offset, 2)
            x = gutter + col * (thumb_width + gutter)
            y = gutter + row * (cell_height + gutter)
            sheet.paste(image, (x, y + label_height))
            draw.text((x, y + 10), f"Trang {start + offset + 1}", fill="black")
        output = CONTACTS / f"contact-{sheet_index:02d}.png"
        sheet.save(output, "PNG")


def main():
    paths = render_pages()
    make_contacts(paths)
    print(f"Rendered {len(paths)} pages")
    print(OUT)
    print(CONTACTS)


if __name__ == "__main__":
    main()
