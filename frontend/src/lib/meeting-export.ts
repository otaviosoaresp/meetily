import { imageDataToDataUrl } from '@/lib/transcript-annotation-input';
import { marked } from 'marked';
import type { TDocumentDefinitions } from 'pdfmake/interfaces';
import type { Transcript, TranscriptAnnotation } from '@/types';

export interface MeetingExportInput {
  title: string;
  date: string;
  summaryMarkdown?: string | null;
  segments: Transcript[];
  annotations: ExportAnnotation[];
}

export type MeetingPdfDefinition = TDocumentDefinitions;

export type ExportAnnotation = TranscriptAnnotation & {
  /** Resolved image data from an annotation file, ready for Markdown/PDF output. */
  imageDataUrl?: string;
};

export type MeetingExportEvent =
  | { kind: 'segment'; time: number; segment: Transcript }
  | { kind: 'annotation'; time: number; annotation: ExportAnnotation };

type MarkedToken = {
  type?: string;
  text?: string;
  raw?: string;
  depth?: number;
  tokens?: MarkedToken[];
  items?: MarkedToken[];
  ordered?: boolean;
  start?: number | string;
  lang?: string;
  header?: Array<{ text: string; tokens?: MarkedToken[] }>;
  rows?: Array<Array<{ text: string; tokens?: MarkedToken[] }>>;
  href?: string;
};

type PdfInline = Record<string, unknown>;

function inlineToPdfContent(tokens: MarkedToken[] = [], inherited: Record<string, unknown> = {}): PdfInline[] {
  return tokens.flatMap(token => {
    const nested = token.tokens ? inlineToPdfContent(token.tokens, inherited) : [];
    switch (token.type) {
      case 'strong':
        return token.tokens ? inlineToPdfContent(token.tokens, { ...inherited, bold: true }) : [];
      case 'em':
        return token.tokens ? inlineToPdfContent(token.tokens, { ...inherited, italics: true }) : [];
      case 'del':
        return token.tokens ? inlineToPdfContent(token.tokens, { ...inherited, decoration: 'lineThrough' }) : [];
      case 'codespan':
        return [{ text: token.text || '', ...inherited, font: 'Roboto Mono' }];
      case 'link':
        return token.tokens
          ? inlineToPdfContent(token.tokens, { ...inherited, color: '#2563eb', decoration: 'underline' })
          : [];
      case 'image':
        return token.href?.startsWith('data:')
          ? [{ image: token.href, width: 480, margin: [0, 4, 0, 8] }]
          : [{ text: token.text || 'Image unavailable', ...inherited }];
      case 'br':
        return [{ text: '\n', ...inherited }];
      case 'text':
      case 'escape':
        return [{ text: token.text || '', ...inherited }];
      default:
        if (nested.length > 0) return nested;
        return token.text || token.raw ? [{ text: token.text || token.raw || '', ...inherited }] : [];
    }
  });
}

function inlineBlock(tokens: MarkedToken[] = [], style?: string): Record<string, unknown> {
  const content = inlineToPdfContent(tokens);
  const hasImage = content.some(item => 'image' in item);
  if (hasImage) return { stack: content, style };
  return { text: content, style };
}

function listItemToPdfContent(item: MarkedToken): Record<string, unknown> {
  const blocks = (item.tokens || []).flatMap(tokenToPdfContent);
  if (blocks.length === 1 && 'text' in blocks[0]) return blocks[0];
  return { stack: blocks };
}

function tableCellToPdfContent(cell: { text: string; tokens?: MarkedToken[] }, header = false): Record<string, unknown> {
  return { ...inlineBlock(cell.tokens || [{ type: 'text', text: cell.text }]), style: header ? 'tableHeader' : 'tableCell' };
}

function tokenToPdfContent(token: MarkedToken): Record<string, unknown>[] {
  switch (token.type) {
    case 'heading':
      return [inlineBlock(token.tokens || [{ type: 'text', text: token.text || '' }], `heading${token.depth || 1}`)];
    case 'paragraph':
      return [inlineBlock(token.tokens || [{ type: 'text', text: token.text || '' }], 'paragraph')];
    case 'list':
      return [{
        [token.ordered ? 'ol' : 'ul']: (token.items || []).map(listItemToPdfContent),
        ...(token.ordered && typeof token.start === 'number' && token.start !== 1 ? { start: token.start } : {}),
        style: 'list',
      }];
    case 'blockquote':
      return [{ stack: (token.tokens || []).flatMap(tokenToPdfContent), style: 'blockquote' }];
    case 'code':
      return [{ text: token.text || '', style: 'code' }];
    case 'table':
      return [{
        table: {
          headerRows: 1,
          widths: (token.header || []).map(() => '*'),
          body: [
            (token.header || []).map(cell => tableCellToPdfContent(cell, true)),
            ...(token.rows || []).map(row => row.map(cell => tableCellToPdfContent(cell))),
          ],
        },
        layout: 'lightHorizontalLines',
        margin: [0, 4, 0, 10],
      }];
    case 'hr':
      return [{ canvas: [{ type: 'line', x1: 0, y1: 0, x2: 515, y2: 0, lineWidth: 1, lineColor: '#cbd5e1' }] }];
    case 'space':
      return [];
    default:
      return token.text || token.raw ? [{ text: token.text || token.raw || '', style: 'paragraph' }] : [];
  }
}

/** Convert export Markdown into a native pdfmake document definition. */
export function markdownToPdfDefinition(markdown: string): MeetingPdfDefinition {
  return {
    content: (marked.lexer(markdown, { gfm: true }) as MarkedToken[]).flatMap(tokenToPdfContent),
    defaultStyle: { font: 'Roboto', fontSize: 10, lineHeight: 1.2 },
    styles: {
      heading1: { fontSize: 20, bold: true, margin: [0, 0, 0, 12] },
      heading2: { fontSize: 15, bold: true, margin: [0, 14, 0, 7] },
      heading3: { fontSize: 12, bold: true, margin: [0, 10, 0, 5] },
      heading4: { fontSize: 11, bold: true, margin: [0, 8, 0, 4] },
      heading5: { fontSize: 10, bold: true, margin: [0, 7, 0, 3] },
      heading6: { fontSize: 10, bold: true, margin: [0, 6, 0, 3] },
      paragraph: { margin: [0, 0, 0, 7] },
      list: { margin: [0, 0, 0, 7] },
      blockquote: { margin: [10, 4, 0, 10], color: '#475569', italics: true },
      code: { font: 'Roboto Mono', fontSize: 8, background: '#f1f5f9', margin: [0, 4, 0, 10] },
      tableHeader: { bold: true, fillColor: '#e2e8f0' },
      tableCell: { margin: [2, 2, 2, 2] },
    },
  } as unknown as MeetingPdfDefinition;
}

function timestamp(seconds: number | undefined): string {
  if (seconds === undefined || !Number.isFinite(seconds)) return '[--:--]';
  const totalSeconds = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = totalSeconds % 60;
  return `[${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}]`;
}

function eventTime(event: MeetingExportEvent): number {
  return event.kind === 'segment'
    ? event.segment.audio_start_time ?? Number.POSITIVE_INFINITY
    : event.annotation.anchorTime;
}

/** Merge transcript segments and annotations in recording-relative time order. */
export function buildExportTimeline(
  segments: Transcript[],
  annotations: ExportAnnotation[],
): MeetingExportEvent[] {
  const events: Array<MeetingExportEvent & { originalIndex: number }> = [
    ...segments.map((segment, originalIndex) => ({
      kind: 'segment' as const,
      time: segment.audio_start_time ?? Number.POSITIVE_INFINITY,
      segment,
      originalIndex,
    })),
    ...annotations.map((annotation, originalIndex) => ({
      kind: 'annotation' as const,
      time: annotation.anchorTime,
      annotation,
      originalIndex: segments.length + originalIndex,
    })),
  ];

  return events
    .sort((a, b) => {
      const timeDifference = eventTime(a) - eventTime(b);
      if (timeDifference !== 0) return timeDifference;
      // Annotations at a segment's timestamp appear before that segment's text.
      if (a.kind !== b.kind) return a.kind === 'annotation' ? -1 : 1;
      return a.originalIndex - b.originalIndex;
    })
    .map(event => event.kind === 'segment'
      ? { kind: event.kind, time: event.time, segment: event.segment }
      : { kind: event.kind, time: event.time, annotation: event.annotation });
}

function annotationImageDataUrl(annotation: ExportAnnotation): string | null {
  if (annotation.imageDataUrl?.startsWith('data:')) return annotation.imageDataUrl;
  if (annotation.imageData && annotation.imageData.length > 0) {
    return imageDataToDataUrl(annotation.imageData, annotation.imageMime || 'image/png');
  }
  return null;
}

function renderAnnotation(annotation: ExportAnnotation): string {
  const time = timestamp(annotation.anchorTime);
  if (annotation.type === 'image') {
    const imageDataUrl = annotationImageDataUrl(annotation);
    if (!imageDataUrl) return `- ${time} Image: ${annotation.text?.trim() || 'Image unavailable'}`;
    const caption = annotation.text?.trim() || 'Screenshot';
    return `![${caption} ${time}](${imageDataUrl})`;
  }

  const label = annotation.type === 'bookmark' ? 'Bookmark' : 'Note';
  const text = annotation.text?.trim();
  return `- ${time} ${label}${text ? `: ${text}` : ''}`;
}

function renderSegment(segment: Transcript): string {
  const source = segment.source && segment.source !== 'Audio' ? `${segment.source}: ` : '';
  return `${timestamp(segment.audio_start_time)} ${source}${segment.text}`;
}

/** Build the portable Markdown representation of a saved meeting. */
export function buildMeetingMarkdown({
  title,
  date,
  summaryMarkdown,
  segments,
  annotations,
}: MeetingExportInput): string {
  const parts = [`# ${title}`, `## Date: ${date}`];
  if (summaryMarkdown?.trim()) {
    parts.push(`## AI Summary\n\n${summaryMarkdown.trim()}`);
  }

  parts.push('## Transcript');
  parts.push(
    ...buildExportTimeline(segments, annotations).map(event =>
      event.kind === 'segment' ? renderSegment(event.segment) : renderAnnotation(event.annotation)
    ),
  );

  return `${parts.join('\n\n')}\n`;
}

/** Resolve the image source used by both Markdown and PDF exports. */
export function getExportImageDataUrl(annotation: ExportAnnotation): string | null {
  return annotationImageDataUrl(annotation);
}
