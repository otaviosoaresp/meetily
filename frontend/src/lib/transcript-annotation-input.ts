import type { NewTranscriptAnnotation, TranscriptSegmentData } from '@/types';

export interface ClipboardImage {
  rgba: () => Promise<Uint8Array>;
  size: () => Promise<{ width: number; height: number }>;
}

export type ClipboardImageReader = () => Promise<ClipboardImage>;
export type PngEncoder = (
  rgba: Uint8Array,
  size: { width: number; height: number },
) => Promise<Uint8Array>;

function formatAnchorTime(seconds: number): string {
  const totalSeconds = Math.floor(seconds);
  const minutes = Math.floor(totalSeconds / 60);
  const remainingSeconds = totalSeconds % 60;
  return `[${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}]`;
}

/** Resolve the point at which a manual annotation will be inserted. */
export function resolveTranscriptAnchor(
  activeTimestamp: number | null,
  latestSegment?: Pick<TranscriptSegmentData, 'timestamp' | 'endTime'>,
) {
  const time = activeTimestamp ?? latestSegment?.endTime ?? latestSegment?.timestamp ?? 0;
  const isDefault = activeTimestamp === null;
  return {
    time,
    isDefault,
    label: `${formatAnchorTime(time)} ${isDefault ? 'latest transcript point (default)' : 'selected transcript point'}`,
  };
}

/** Encode the Tauri Image RGBA resource as a portable PNG for annotation storage. */
export const encodeRgbaAsPng: PngEncoder = async (rgba, size) => {
  if (size.width <= 0 || size.height <= 0 || rgba.length !== size.width * size.height * 4) {
    throw new Error('Clipboard image has invalid dimensions');
  }

  const canvas = document.createElement('canvas');
  canvas.width = size.width;
  canvas.height = size.height;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas is unavailable for clipboard image encoding');

  context.putImageData(new ImageData(new Uint8ClampedArray(rgba), size.width, size.height), 0, 0);
  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(result => {
      if (result) resolve(result);
      else reject(new Error('Failed to encode clipboard image as PNG'));
    }, 'image/png');
  });
  return new Uint8Array(await blob.arrayBuffer());
};

/** Read a native clipboard image and turn it into the existing annotation payload. */
export async function readClipboardImageAnnotation(
  readImage: ClipboardImageReader,
  anchorTime: number,
  encodePng: PngEncoder = encodeRgbaAsPng,
): Promise<NewTranscriptAnnotation | null> {
  try {
    const image = await readImage();
    const [rgba, size] = await Promise.all([image.rgba(), image.size()]);
    const png = await encodePng(rgba, size);
    return {
      type: 'image',
      anchorTime,
      imageData: Array.from(png),
      imageMime: 'image/png',
    };
  } catch {
    return null;
  }
}
