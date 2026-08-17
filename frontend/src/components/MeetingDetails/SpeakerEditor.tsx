"use client";

import { useEffect, useState } from "react";
import { Speaker } from "@/types";

interface SpeakerEditorProps {
  speakers: Speaker[];
  onRename: (speakerId: number, name: string) => Promise<void>;
}

export function SpeakerEditor({ speakers, onRename }: SpeakerEditorProps) {
  const [draftNames, setDraftNames] = useState<Record<number, string>>({});
  const [savingSpeakerId, setSavingSpeakerId] = useState<number | null>(null);

  useEffect(() => {
    setDraftNames(Object.fromEntries(
      speakers.map(speaker => [speaker.speaker_id, speaker.name ?? ""]),
    ));
  }, [speakers]);

  if (speakers.length === 0) return null;

  return (
    <div className="border-b border-gray-200 px-4 py-3">
      <p className="mb-2 text-xs font-medium text-gray-600">Speakers</p>
      <div className="space-y-2">
        {speakers.map(speaker => {
          const defaultLabel = `Outros ${speaker.speaker_id + 1}`;
          const draft = draftNames[speaker.speaker_id] ?? "";
          const isSaving = savingSpeakerId === speaker.speaker_id;

          return (
            <div key={speaker.speaker_id} className="flex items-center gap-2">
              <span className="min-w-16 text-xs text-gray-500">{defaultLabel}</span>
              <input
                className="min-w-0 flex-1 rounded border border-gray-300 px-2 py-1 text-xs"
                value={draft}
                placeholder={defaultLabel}
                onChange={event => setDraftNames(prev => ({
                  ...prev,
                  [speaker.speaker_id]: event.target.value,
                }))}
                aria-label={`Name for ${defaultLabel}`}
              />
              <button
                type="button"
                className="rounded bg-gray-100 px-2 py-1 text-xs text-gray-700 disabled:opacity-50"
                disabled={isSaving}
                onClick={async () => {
                  setSavingSpeakerId(speaker.speaker_id);
                  try {
                    await onRename(speaker.speaker_id, draft.trim());
                  } finally {
                    setSavingSpeakerId(null);
                  }
                }}
              >
                {isSaving ? "Saving…" : "Save"}
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
