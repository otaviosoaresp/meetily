import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import ts from 'typescript';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const requireModule = createRequire(import.meta.url);

const modulePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  '..', '..', 'src', 'lib', 'meeting-export.ts'
);
const source = fs.readFileSync(modulePath, 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const cjsModule = { exports: {} };
vm.runInNewContext(compiled, {
  exports: cjsModule.exports,
  module: cjsModule,
  require: moduleName => {
    if (moduleName === '@/lib/transcript-annotation-input') {
      return {
        imageDataToDataUrl: (imageData, mime = 'image/png') =>
          `data:${mime};base64,${Buffer.from(Uint8Array.from(imageData)).toString('base64')}`,
      };
    }
    if (moduleName === 'marked') return requireModule('marked');
    throw new Error(`Unexpected module: ${moduleName}`);
  },
  btoa: value => Buffer.from(value, 'binary').toString('base64'),
});

const { buildMeetingMarkdown, markdownToPdfDefinition } = cjsModule.exports;
const markdown = buildMeetingMarkdown({
  title: 'Planning & Review',
  date: 'August 31, 2026',
  summaryMarkdown: 'The team agreed to ship the export flow.',
  segments: [
    { id: 'second', text: 'The follow-up starts now.', audio_start_time: 12, source: 'Outros' },
    { id: 'first', text: 'We reviewed the plan.', audio_start_time: 4, source: 'Você' },
  ],
  annotations: [
    { id: 'note-1', type: 'note', anchorTime: 6, text: 'Confirm the PDF layout.', createdAt: '2026-08-31T20:00:00.000Z' },
    { id: 'image-1', type: 'image', anchorTime: 14, text: 'Whiteboard', imageData: [137, 80, 78, 71], imageMime: 'image/png', createdAt: '2026-08-31T20:01:00.000Z' },
  ],
});

assert.match(markdown, /^# Planning & Review\n\n## Date: August 31, 2026/m);
assert.match(markdown, /## AI Summary\n\nThe team agreed to ship the export flow\./);
assert.ok(markdown.indexOf('[00:04] Você: We reviewed the plan.') < markdown.indexOf('[00:06] Note: Confirm the PDF layout.'));
assert.ok(markdown.indexOf('[00:06] Note: Confirm the PDF layout.') < markdown.indexOf('[00:12] Outros: The follow-up starts now.'));
assert.match(markdown, /!\[Whiteboard \[00:14\]\]\(data:image\/png;base64,iVBORw==\)/);

const pdfDefinition = markdownToPdfDefinition(`${markdown}\n\n| Owner | Status |\n| --- | --- |\n| Captain | Ready |`);
const table = pdfDefinition.content.find(node => node.table);
assert.ok(table, 'PDF definition should contain a native table node');
assert.equal(table.table.headerRows, 1);
assert.deepEqual(table.table.body[0].map(cell => cell.text[0].text), ['Owner', 'Status']);
assert.deepEqual(table.table.body[1].map(cell => cell.text[0].text), ['Captain', 'Ready']);
assert.ok(JSON.stringify(pdfDefinition).includes('data:image/png;base64,iVBORw=='));

const listDefinition = markdownToPdfDefinition('- **Owner:** Captain decided to ship\n- See `ops-oncall` and [the doc](https://example.com)');
const list = listDefinition.content.find(node => node.ul);
assert.ok(list, 'PDF definition should contain a native unordered list');
assert.ok(Array.isArray(list.ul[0].text), 'list item should preserve inline token content');
assert.equal(list.ul[0].text[0].text, 'Owner:');
assert.equal(list.ul[0].text[0].bold, true);
assert.equal(list.ul[0].text[1].text, ' Captain decided to ship');

const unsupportedImageDefinition = markdownToPdfDefinition('![WebP](data:image/webp;base64,UklGRg==)');
assert.match(JSON.stringify(unsupportedImageDefinition), /Image unavailable/);
assert.equal(JSON.stringify(unsupportedImageDefinition).includes('"image"'), false);

const validPng = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=';
const imageDefinition = markdownToPdfDefinition(`![Tall screenshot](${validPng})`);
const imageNode = imageDefinition.content[0].stack.find(node => node.image);
assert.equal(imageNode.fit[0], 480);
assert.equal(imageNode.fit[1], 700);

const pdfMake = requireModule('pdfmake/build/pdfmake');
const pdfFonts = requireModule('pdfmake/build/vfs_fonts');
pdfMake.addVirtualFileSystem(pdfFonts);
pdfMake.addFonts({ 'Roboto Mono': { normal: 'Roboto-Regular.ttf' } });
const pdfBuffer = await pdfMake.createPdf(markdownToPdfDefinition(`![Screenshot](${validPng})\n\n![WebP](data:image/webp;base64,UklGRg==)`)).getBuffer();
assert.ok(pdfBuffer.length > 0, 'unsupported image formats should not abort PDF generation');

console.log('meeting export tests passed');
