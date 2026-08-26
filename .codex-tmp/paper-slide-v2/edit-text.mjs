import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const starterPath = 'E:\\SCT_Camera\\.codex-tmp\\paper-slide-v2\\template-starter.pptx';
const finalPath = 'E:\\Slide_Cong_bo_khoa_hoc_FDSE_v2.pptx';
const tmpDir = 'E:\\SCT_Camera\\.codex-tmp\\paper-slide-v2';
const finalLayoutDir = `${tmpDir}\\final-layout\\final`;

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

const presentation = await PresentationFile.importPptx(await FileBlob.load(starterPath));
const slide = presentation.slides.getItem(0);
const before = await presentation.inspect({
  kind: 'slide,textbox,shape,image,notes,layout',
  maxChars: 30000,
});
await fs.writeFile(`${tmpDir}\\before-inspect.ndjson`, before.ndjson, 'utf8');
await writeBlob(
  `${tmpDir}\\before-slide.png`,
  await presentation.export({ slide, format: 'png', scale: 1 }),
);

const records = before.ndjson
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));
const byName = new Map(
  records
    .filter((record) => record.name && record.id)
    .map((record) => [record.name, record.id]),
);

function shape(name) {
  const id = byName.get(name);
  if (!id) throw new Error(`Missing inherited shape: ${name}`);
  return presentation.resolve(id);
}

shape('TextBox 33').text.replace('ACCEPTED', 'ĐÃ CHẤP NHẬN');
shape('TextBox 61').text.replace('Được chấp nhận tại FDSE', 'Hội nghị quốc tế FDSE');

const body = shape('TextBox 62');
body.position.merge({ left: 130, top: 455, width: 570, height: 120 });
body.text.style = { fontSize: 30 };

const callout = shape('TextBox 63');
callout.text.replace(
  'Được chấp nhận đăng trong kỷ yếu do Springer xuất bản.',
  'Đăng trong kỷ yếu do Springer xuất bản.',
);
callout.position.merge({ left: 130, top: 585, width: 570, height: 90 });
callout.text.style = { fontSize: 30 };

await fs.mkdir(finalLayoutDir, { recursive: true });
await writeBlob(
  `${tmpDir}\\after-slide.png`,
  await presentation.export({ slide, format: 'png', scale: 1 }),
);
const layout = await slide.export({ format: 'layout' });
await fs.writeFile(`${finalLayoutDir}\\slide-01.layout.json`, await layout.text());

const after = await presentation.inspect({
  kind: 'slide,textbox,shape,image,notes,layout',
  maxChars: 30000,
});
await fs.writeFile(`${tmpDir}\\after-inspect.ndjson`, after.ndjson, 'utf8');

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(finalPath);

