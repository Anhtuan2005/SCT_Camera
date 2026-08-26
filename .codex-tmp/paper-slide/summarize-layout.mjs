import fs from 'node:fs/promises';

const layout = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
for (const element of layout.elements ?? []) {
  console.log(JSON.stringify({
    order: element.order,
    aid: element.aid,
    name: element.name,
    kind: element.kind,
    bbox: element.bbox,
    text: element.text,
    image: element.image?.assetId ?? element.fillImage?.assetId,
  }));
}
