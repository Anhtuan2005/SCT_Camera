import fs from 'node:fs/promises';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const starterPath = 'E:\\SCT_Camera\\.codex-tmp\\paper-slide\\template-starter.pptx';
const imagePath = 'C:\\Users\\Admin\\AppData\\Local\\Temp\\codex-clipboard-b72f4f2a-77c1-41c0-ad7f-fe663b503945.png';
const finalPath = 'E:\\Slide_Cong_bo_khoa_hoc_FDSE.pptx';
const previewPath = 'E:\\SCT_Camera\\.codex-tmp\\paper-slide\\final-slide.png';
const layoutDir = 'E:\\SCT_Camera\\.codex-tmp\\paper-slide\\final-layout\\final';

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

const presentation = await PresentationFile.importPptx(await FileBlob.load(starterPath));
const slide = presentation.slides.getItem(0);
const initialSnapshot = await presentation.inspect({
  kind: 'slide,textbox,shape,notes',
  maxChars: 30000,
});
const initialRecords = initialSnapshot.ndjson
  .split(/\r?\n/)
  .filter(Boolean)
  .map((line) => JSON.parse(line));
const byName = new Map(
  initialRecords
    .filter((record) => record.name && record.id)
    .map((record) => [record.name, record.id]),
);
const notesRecord = initialRecords.find((record) => record.kind === 'notes');

function resolveName(name) {
  const id = byName.get(name);
  if (!id) throw new Error(`Missing inherited shape: ${name}`);
  return presentation.resolve(id);
}

function rewrite(name, oldText, newText, position) {
  const shape = resolveName(name);
  shape.text.replace(oldText, newText);
  if (position) shape.position.merge(position);
  return shape;
}

function move(name, position) {
  resolveName(name).position.merge(position);
}

function remove(name) {
  resolveName(name).delete();
}

rewrite(
  'TextBox 18',
  'CHƯƠNG 4 · KIỂM THỬ NGHIỆP VỤ CHI TIẾT',
  'CHƯƠNG 4 · KẾT QUẢ THỰC NGHIỆM',
);
rewrite(
  'TextBox 21',
  'Minh họa kịch bản kiểm thử (2/2)',
  'Kết quả công bố khoa học',
);
rewrite(
  'TextBox 12',
  'Chương 4 · Kiểm thử nghiệp vụ chi tiết',
  'Chương 4 · Kết quả đạt được',
);
rewrite('TextBox 15', '17', '20');

move('Freeform 26', { left: 780, top: 265, width: 1010, height: 465 });
resolveName('Freeform 26').fill = 'none';
move('Freeform 30', { left: 130, top: 250, width: 175, height: 42 });
move('Freeform 32', { left: 130, top: 250, width: 175, height: 42 });
rewrite('TextBox 33', 'PASSED', 'ACCEPTED', {
  left: 130,
  top: 248.5,
  width: 175,
  height: 43.5,
});

rewrite('TextBox 36', 'TC-05 · TC-E2E-05', 'Tác giả: Hạnh Nguyễn · Nguyễn Nguyễn · Tuấn Nguyễn', {
  left: 810,
  top: 693,
  width: 950,
  height: 45,
});

rewrite('TextBox 61', 'Người lạ xuất hiện', 'Được chấp nhận tại FDSE', {
  left: 130,
  top: 335,
  width: 570,
  height: 95,
});
rewrite(
  'TextBox 62',
  'Cosine similarity dưới ngưỡng nhận diện → gắn nhãn Stranger tự động.',
  'Bài báo phát triển từ kết quả nghiên cứu của khóa luận.',
  { left: 130, top: 465, width: 570, height: 145 },
);
rewrite(
  'TextBox 63',
  'PASSED - STRANGER DETECTED được gắn nhãn.',
  'Được chấp nhận đăng trong kỷ yếu do Springer xuất bản.',
  { left: 130, top: 650, width: 570, height: 120 },
);

for (const id of [
  'Freeform 28',
  'Freeform 35',
  'Freeform 38',
  'Freeform 40',
  'Freeform 42',
  'Freeform 44',
  'TextBox 45',
  'Freeform 47',
  'TextBox 48',
  'Freeform 50',
  'Freeform 52',
  'Freeform 54',
  'Freeform 56',
  'TextBox 57',
  'Freeform 59',
  'TextBox 60',
  'TextBox 64',
  'TextBox 65',
  'TextBox 66',
  'TextBox 67',
  'TextBox 68',
  'TextBox 69',
]) remove(id);

const imageBytes = await fs.readFile(imagePath);
slide.images.add({
  blob: new Uint8Array(imageBytes),
  contentType: 'image/png',
  alt: 'Trang đầu bài báo Risk-Aware Intelligent Video Surveillance Using Multi-Object Tracking and Behavior Analytics',
  fit: 'contain',
  position: { left: 810, top: 310, width: 950, height: 350 },
  geometry: 'rect',
});

if (!notesRecord?.id) throw new Error('Missing inherited speaker notes.');
const notes = presentation.resolve(notesRecord.id);
notes.setText([
  '[Sources]',
  '- User-provided paper header screenshot: C:\\Users\\Admin\\AppData\\Local\\Temp\\codex-clipboard-b72f4f2a-77c1-41c0-ad7f-fe663b503945.png',
  '- User-provided publication status: accepted at FDSE; accepted for proceedings published by Springer.',
  '[/Sources]',
].join('\n'));

await fs.mkdir(layoutDir, { recursive: true });
await writeBlob(previewPath, await presentation.export({ slide, format: 'png', scale: 1 }));
const layout = await slide.export({ format: 'layout' });
await fs.writeFile(`${layoutDir}\\slide-01.layout.json`, await layout.text());

const snapshot = await presentation.inspect({
  kind: 'slide,textbox,shape,image,notes,layout',
  maxChars: 30000,
});
await fs.writeFile(
  'E:\\SCT_Camera\\.codex-tmp\\paper-slide\\final-inspect.ndjson',
  snapshot.ndjson,
  'utf8',
);

const pptx = await PresentationFile.exportPptx(presentation);
await pptx.save(finalPath);
