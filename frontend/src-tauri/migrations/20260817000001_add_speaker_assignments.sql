-- Meeting-local speaker clusters and manual transcript corrections.
-- Both additions are nullable/backward compatible with existing meetings.

ALTER TABLE transcripts ADD COLUMN speaker_id INTEGER;

CREATE TABLE IF NOT EXISTS meeting_speakers (
    meeting_id TEXT NOT NULL,
    speaker_id INTEGER NOT NULL,
    name TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (meeting_id, speaker_id),
    FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_meeting_speakers_meeting_id
    ON meeting_speakers(meeting_id);
