import { useCallback, useState, type RefObject } from 'react';
import { save } from '@tauri-apps/plugin-dialog';
import { writeFile } from '@tauri-apps/plugin-fs';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'sonner';
import type { Summary, Transcript, TranscriptAnnotation } from '@/types';
import type { BlockNoteSummaryViewRef } from '@/components/AISummary/BlockNoteSummaryView';
import {
  buildMeetingMarkdown,
  getExportImageDataUrl,
  markdownToPdfDefinition,
  type ExportAnnotation,
} from '@/lib/meeting-export';

interface UseMeetingExportProps {
  meeting: { id: string; title: string; created_at: string };
  meetingTitle: string;
  aiSummary: Summary | null;
  summaryRef: RefObject<BlockNoteSummaryViewRef>;
  annotations: TranscriptAnnotation[];
  getAnnotationImage?: (annotation: TranscriptAnnotation) => Promise<string | null>;
}

interface TranscriptPage {
  transcripts: Transcript[];
  total_count: number;
}

function safeFileName(value: string): string {
  return (value || 'meeting-export').replace(/[\\/:*?"<>|]/g, '-').trim() || 'meeting-export';
}

function withExtension(filePath: string, extension: string): string {
  return filePath.toLowerCase().endsWith(`.${extension}`) ? filePath : `${filePath}.${extension}`;
}

function legacySummaryToMarkdown(summary: Summary): string {
  return Object.entries(summary)
    .filter(([key, section]) =>
      !['markdown', 'summary_json', '_section_order', 'MeetingName'].includes(key) &&
      section && typeof section === 'object' && 'title' in section && 'blocks' in section
    )
    .map(([, section]) => {
      const typedSection = section as { title?: string; blocks?: Array<{ content?: string; type?: string }> };
      const content = (typedSection.blocks || []).map(block => {
        if (block.type === 'heading1') return `### ${block.content || ''}`;
        if (block.type === 'heading2') return `#### ${block.content || ''}`;
        if (block.type === 'bullet') return `- ${block.content || ''}`;
        return block.content || '';
      }).join('\n');
      return `## ${typedSection.title || 'Summary'}\n\n${content}`;
    })
    .filter(section => section.trim())
    .join('\n\n');
}

async function renderMarkdownPdf(markdown: string): Promise<Uint8Array> {
  const [{ default: pdfMake }, { default: pdfFonts }] = await Promise.all([
    import('pdfmake/build/pdfmake'),
    import('pdfmake/build/vfs_fonts'),
  ]);
  pdfMake.addVirtualFileSystem(pdfFonts);
  pdfMake.addFonts({
    'Roboto Mono': {
      normal: 'Roboto-Regular.ttf',
      bold: 'Roboto-Medium.ttf',
      italics: 'Roboto-Italic.ttf',
      bolditalics: 'Roboto-MediumItalic.ttf',
    },
  });
  const buffer = await pdfMake.createPdf(markdownToPdfDefinition(markdown)).getBuffer();
  return new Uint8Array(buffer);
}

export function useMeetingExport({
  meeting,
  meetingTitle,
  aiSummary,
  summaryRef,
  annotations,
  getAnnotationImage,
}: UseMeetingExportProps) {
  const [isExporting, setIsExporting] = useState(false);

  const fetchAllTranscripts = useCallback(async (): Promise<Transcript[]> => {
    const firstPage = await invoke<TranscriptPage>('api_get_meeting_transcripts', {
      meetingId: meeting.id,
      limit: 1,
      offset: 0,
    });
    if (firstPage.total_count === 0) return [];
    const allData = await invoke<TranscriptPage>('api_get_meeting_transcripts', {
      meetingId: meeting.id,
      limit: firstPage.total_count,
      offset: 0,
    });
    return allData.transcripts;
  }, [meeting.id]);

  const getSummaryMarkdown = useCallback(async (): Promise<string> => {
    if (summaryRef.current?.getMarkdown) {
      const editorMarkdown = await summaryRef.current.getMarkdown();
      if (editorMarkdown.trim()) return editorMarkdown;
    }
    if (!aiSummary) return '';
    if ('markdown' in aiSummary && typeof (aiSummary as Summary & { markdown?: unknown }).markdown === 'string') {
      return (aiSummary as Summary & { markdown: string }).markdown;
    }
    return legacySummaryToMarkdown(aiSummary);
  }, [aiSummary, summaryRef]);

  const prepareExport = useCallback(async () => {
    const [transcripts, summaryMarkdown] = await Promise.all([
      fetchAllTranscripts(),
      getSummaryMarkdown(),
    ]);
    const resolvedAnnotations: ExportAnnotation[] = await Promise.all(annotations.map(async annotation => {
      if (annotation.type !== 'image') return annotation;
      const dataUrl = getExportImageDataUrl(annotation);
      if (dataUrl) return { ...annotation, imageDataUrl: dataUrl };
      if (annotation.imageFile && getAnnotationImage) {
        return { ...annotation, imageDataUrl: await getAnnotationImage(annotation) || undefined };
      }
      return annotation;
    }));
    const markdown = buildMeetingMarkdown({
      title: meetingTitle || meeting.title,
      date: new Date(meeting.created_at).toLocaleDateString(),
      summaryMarkdown,
      segments: transcripts,
      annotations: resolvedAnnotations,
    });
    return { markdown, resolvedAnnotations };
  }, [annotations, fetchAllTranscripts, getAnnotationImage, getSummaryMarkdown, meeting.created_at, meeting.title, meetingTitle]);

  const runExport = useCallback(async (operation: () => Promise<void>, failureMessage: string) => {
    setIsExporting(true);
    try {
      await operation();
    } catch (error) {
      console.error(failureMessage, error);
      toast.error(failureMessage);
    } finally {
      setIsExporting(false);
    }
  }, []);

  const copyAsMarkdown = useCallback(() => runExport(async () => {
    const { markdown } = await prepareExport();
    await navigator.clipboard.writeText(markdown);
    toast.success('Meeting Markdown copied to clipboard');
  }, 'Failed to copy meeting Markdown'), [prepareExport, runExport]);

  const exportMarkdown = useCallback(() => runExport(async () => {
    const destination = await save({
      defaultPath: `${safeFileName(meetingTitle || meeting.title)}.md`,
      filters: [{ name: 'Markdown', extensions: ['md'] }],
    });
    if (!destination) return;
    const { markdown } = await prepareExport();
    await writeFile(withExtension(destination, 'md'), new TextEncoder().encode(markdown));
    toast.success('Meeting exported as Markdown');
  }, 'Failed to export meeting Markdown'), [meeting.title, meetingTitle, prepareExport, runExport]);

  const exportPdf = useCallback(() => runExport(async () => {
    const destination = await save({
      defaultPath: `${safeFileName(meetingTitle || meeting.title)}.pdf`,
      filters: [{ name: 'PDF', extensions: ['pdf'] }],
    });
    if (!destination) return;
    const { markdown } = await prepareExport();
    await writeFile(withExtension(destination, 'pdf'), await renderMarkdownPdf(markdown));
    toast.success('Meeting exported as PDF');
  }, 'Failed to export meeting PDF'), [meeting.title, meetingTitle, prepareExport, runExport]);

  return { copyAsMarkdown, exportMarkdown, exportPdf, isExporting };
}
