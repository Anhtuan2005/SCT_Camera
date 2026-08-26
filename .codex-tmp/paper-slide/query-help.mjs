import { FileBlob, PresentationFile } from '@oai/artifact-tool';

const presentation = await PresentationFile.importPptx(
  await FileBlob.load('E:\\SCT_Camera\\.codex-tmp\\paper-slide\\template-starter.pptx'),
);
const result = presentation.help('*', {
  search: 'slide.shapes.delete|shape.delete|slide.shapes.add|deleteAll|shape position',
  include: ['index', 'examples', 'notes'],
  maxChars: 12000,
});
console.log(result.ndjson);
