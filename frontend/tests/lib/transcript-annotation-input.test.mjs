import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import ts from 'typescript';
import { fileURLToPath } from 'node:url';

const modulePath = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  '..', '..', 'src', 'lib', 'transcript-annotation-input.ts'
);
const source = fs.readFileSync(modulePath, 'utf8');
const compiled = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const cjsModule = { exports: {} };
vm.runInNewContext(compiled, { exports: cjsModule.exports, module: cjsModule });
const { readClipboardImageAnnotation, resolveTranscriptAnchor } = cjsModule.exports;
const componentSource = fs.readFileSync(
  path.join(path.dirname(modulePath), '..', 'components', 'VirtualizedTranscriptView.tsx'),
  'utf8'
);

assert.match(componentSource, /readClipboardImageAnnotation\(readImage, anchorTime\)/, 'the transcript view must read system clipboard images');
assert.match(componentSource, /onKeyDown=\{handleKeyDown\}/, 'Ctrl/Cmd+V must trigger the native image path');
assert.match(componentSource, /aria-current=\{isActive \? 'true' : undefined\}/, 'the selected transcript anchor must be visually identifiable');

const encodedPng = await readClipboardImageAnnotation(
  async () => ({
    rgba: async () => new Uint8Array([255, 0, 0, 255]),
    size: async () => ({ width: 1, height: 1 }),
  }),
  12.5,
  async (rgba, size) => {
    assert.deepEqual(Array.from(rgba), [255, 0, 0, 255]);
    assert.deepEqual(size, { width: 1, height: 1 });
    return new Uint8Array([137, 80, 78, 71]);
  }
);
assert.deepEqual(JSON.parse(JSON.stringify(encodedPng)), {
  type: 'image',
  anchorTime: 12.5,
  imageData: [137, 80, 78, 71],
  imageMime: 'image/png',
}, 'clipboard image reads must become a PNG annotation at the selected anchor');

assert.equal(
  await readClipboardImageAnnotation(async () => { throw new Error('clipboard has no image'); }, 12.5),
  null,
  'text-only clipboards must not create an image annotation'
);

assert.deepEqual(
  JSON.parse(JSON.stringify(resolveTranscriptAnchor(42, { timestamp: 10, endTime: 12 }))),
  { time: 42, isDefault: false, label: '[00:42] selected transcript point' },
  'an explicit transcript selection is the annotation anchor'
);
assert.deepEqual(
  JSON.parse(JSON.stringify(resolveTranscriptAnchor(null, { timestamp: 10, endTime: 12 }))),
  { time: 12, isDefault: true, label: '[00:12] latest transcript point (default)' },
  'without a selection annotations use the latest transcript point and say so'
);

console.log('transcript-annotation-input tests passed');
