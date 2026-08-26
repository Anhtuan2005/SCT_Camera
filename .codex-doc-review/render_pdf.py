from pathlib import Path
import sys

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


pdf_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

pdf = pdfium.PdfDocument(str(pdf_path))
page_paths = []
for index in range(len(pdf)):
    page = pdf[index]
    image = page.render(scale=120 / 72).to_pil().convert("RGB")
    page_path = out_dir / f"page-{index + 1:03d}.png"
    image.save(page_path, optimize=True)
    page_paths.append(page_path)

sheet_dir = out_dir / "sheets"
sheet_dir.mkdir(exist_ok=True)
margin = 24
label_height = 34
for start in range(0, len(page_paths), 4):
    images = [Image.open(path).convert("RGB") for path in page_paths[start : start + 4]]
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images) + label_height
    sheet = Image.new("RGB", (cell_w * 2 + margin * 3, cell_h * 2 + margin * 3), "#c9c9c9")
    draw = ImageDraw.Draw(sheet)
    for offset, image in enumerate(images):
        row, col = divmod(offset, 2)
        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)
        page_no = start + offset + 1
        draw.rectangle((x, y, x + cell_w, y + label_height), fill="white")
        draw.text((x + 10, y + 8), f"Page {page_no}", fill="black")
        sheet.paste(image, (x, y + label_height))
    sheet_path = sheet_dir / f"sheet-{start // 4 + 1:02d}-p{start + 1:03d}-{min(start + 4, len(page_paths)):03d}.jpg"
    sheet.save(sheet_path, quality=88, optimize=True)

print(f"pages={len(page_paths)}")
print(f"sheets={(len(page_paths) + 3) // 4}")
