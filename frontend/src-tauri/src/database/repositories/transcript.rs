use crate::api::{TranscriptSearchResult, TranscriptSegment};
use chrono::Utc;
use sqlx::{Connection, Error as SqlxError, SqlitePool};
use tracing::{error, info};
use uuid::Uuid;

pub struct TranscriptsRepository;

impl TranscriptsRepository {
    /// Saves a new meeting and its associated transcript segments.
    /// This function uses a transaction to ensure that either both the meeting
    /// and all its transcripts are saved, or none of them are.
    pub async fn save_transcript(
        pool: &SqlitePool,
        meeting_title: &str,
        transcripts: &[TranscriptSegment],
        folder_path: Option<String>,
    ) -> Result<String, SqlxError> {
        let meeting_id = format!("meeting-{}", Uuid::new_v4());

        let mut conn = pool.acquire().await?;
        let mut transaction = conn.begin().await?;

        let now = Utc::now();

        // 1. Create the new meeting
        let result = sqlx::query(
            "INSERT INTO meetings (id, title, created_at, updated_at, folder_path) VALUES (?, ?, ?, ?, ?)",
        )
        .bind(&meeting_id)
        .bind(meeting_title)
        .bind(now)
        .bind(now)
        .bind(&folder_path)
        .execute(&mut *transaction)
        .await;

        if let Err(e) = result {
            error!("Failed to create meeting '{}': {}", meeting_title, e);
            transaction.rollback().await?;
            return Err(e);
        }

        info!("Successfully created meeting with id: {}", meeting_id);

        // 2. Save each transcript segment with audio timing fields
        for segment in transcripts {
            let transcript_id = format!("transcript-{}", Uuid::new_v4());
            let result = sqlx::query(
                "INSERT INTO transcripts (id, meeting_id, transcript, timestamp, audio_start_time, audio_end_time, duration, source, speaker_id)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
            )
            .bind(&transcript_id)
            .bind(&meeting_id)
            .bind(&segment.text)
            .bind(&segment.timestamp)
            .bind(segment.audio_start_time)
            .bind(segment.audio_end_time)
            .bind(segment.duration)
            .bind(&segment.source)
            .bind(segment.speaker_id)
            .execute(&mut *transaction)
            .await;

            if let Err(e) = result {
                error!(
                    "Failed to save transcript segment for meeting {}: {}",
                    meeting_id, e
                );
                transaction.rollback().await?;
                return Err(e);
            }
        }

        let mut speaker_ids: Vec<i64> = transcripts
            .iter()
            .filter_map(|segment| segment.speaker_id)
            .collect();
        speaker_ids.sort_unstable();
        speaker_ids.dedup();
        for speaker_id in speaker_ids {
            sqlx::query(
                "INSERT OR IGNORE INTO meeting_speakers (meeting_id, speaker_id, name, created_at, updated_at)
                 VALUES (?, ?, NULL, ?, ?)",
            )
            .bind(&meeting_id)
            .bind(speaker_id)
            .bind(now)
            .bind(now)
            .execute(&mut *transaction)
            .await?;
        }

        info!(
            "Successfully saved {} transcript segments for meeting {}",
            transcripts.len(),
            meeting_id
        );

        // Commit the transaction
        transaction.commit().await?;

        Ok(meeting_id)
    }

    /// Searches for a query string within the transcripts.
    /// It returns a list of matching transcripts with context.
    pub async fn search_transcripts(
        pool: &SqlitePool,
        query: &str,
    ) -> Result<Vec<TranscriptSearchResult>, SqlxError> {
        if query.trim().is_empty() {
            return Ok(Vec::new());
        }

        let search_query = format!("%{}%", query.to_lowercase());

        let rows = sqlx::query_as::<_, (String, String, String, String)>(
            "SELECT m.id, m.title, t.transcript, t.timestamp
             FROM meetings m
             JOIN transcripts t ON m.id = t.meeting_id
             WHERE LOWER(t.transcript) LIKE ?",
        )
        .bind(&search_query)
        .fetch_all(pool)
        .await?;

        let results = rows
            .into_iter()
            .map(|(id, title, transcript, timestamp)| {
                let match_context = Self::get_match_context(&transcript, query);
                TranscriptSearchResult {
                    id,
                    title,
                    match_context,
                    timestamp,
                }
            })
            .collect();

        Ok(results)
    }

    /// Helper function to extract a snippet of text around the first match of a query.
    fn get_match_context(transcript: &str, query: &str) -> String {
        let transcript_lower = transcript.to_lowercase();
        let query_lower = query.to_lowercase();

        match transcript_lower.find(&query_lower) {
            Some(match_index) => {
                let start_index = match_index.saturating_sub(100);
                let end_index = (match_index + query.len() + 100).min(transcript.len());

                let mut context = String::new();
                if start_index > 0 {
                    context.push_str("...");
                }
                context.push_str(&transcript[start_index..end_index]);
                if end_index < transcript.len() {
                    context.push_str("...");
                }
                context
            }
            None => transcript.chars().take(200).collect(), // Fallback to the start of the transcript
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::models::Transcript;
    use crate::database::repositories::speaker::SpeakersRepository;
    use sqlx::sqlite::SqlitePoolOptions;

    const SOURCE_MIGRATION_VERSION: i64 = 20260817000000;
    const SPEAKER_MIGRATION_VERSION: i64 = 20260817000001;

    async fn in_memory_pool() -> SqlitePool {
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .expect("failed to open in-memory sqlite")
    }

    fn segment(id: &str, text: &str, start: f64, source: Option<&str>) -> TranscriptSegment {
        TranscriptSegment {
            id: id.to_string(),
            text: text.to_string(),
            timestamp: "2026-08-17T10:00:00Z".to_string(),
            audio_start_time: Some(start),
            audio_end_time: Some(start + 1.0),
            duration: Some(1.0),
            source: source.map(|s| s.to_string()),
            speaker_id: None,
        }
    }

    #[tokio::test]
    async fn migrated_db_persists_and_reloads_channel_sources() {
        let pool = in_memory_pool().await;
        sqlx::migrate!("./migrations")
            .run(&pool)
            .await
            .expect("migrations must apply to a fresh database");

        let mut system_segment = segment("seg-sys", "fala do sistema", 1.0, Some("Outros"));
        system_segment.speaker_id = Some(7);
        let segments = vec![
            segment("seg-mic", "fala do microfone", 0.0, Some("Você")),
            system_segment,
            segment("seg-legacy", "segmento antigo", 2.0, None),
        ];
        let meeting_id =
            TranscriptsRepository::save_transcript(&pool, "Reunião de teste", &segments, None)
                .await
                .expect("save_transcript must succeed");

        let rows = sqlx::query_as::<_, Transcript>(
            "SELECT * FROM transcripts WHERE meeting_id = ? ORDER BY audio_start_time",
        )
        .bind(&meeting_id)
        .fetch_all(&pool)
        .await
        .expect("reload transcripts");

        let sources: Vec<Option<String>> = rows.into_iter().map(|t| t.source).collect();
        assert_eq!(
            sources,
            vec![
                Some("Você".to_string()),
                Some("Outros".to_string()),
                None,
            ],
            "microphone/system labels must round-trip through SQLite and legacy segments stay NULL"
        );

        let persisted_speaker: Option<i64> = sqlx::query_scalar(
            "SELECT speaker_id FROM transcripts WHERE meeting_id = ? AND transcript = 'fala do sistema'",
        )
        .bind(&meeting_id)
        .fetch_one(&pool)
        .await
        .expect("speaker assignment must round-trip");
        assert_eq!(persisted_speaker, Some(7));

        let speaker_count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM meeting_speakers WHERE meeting_id = ? AND speaker_id = 7",
        )
        .bind(&meeting_id)
        .fetch_one(&pool)
        .await
        .expect("speaker cluster must be persisted");
        assert_eq!(speaker_count, 1);

        let renamed = SpeakersRepository::rename(&pool, &meeting_id, 7, Some("Ana".to_string()))
            .await
            .expect("speaker rename must succeed");
        assert_eq!(renamed.name.as_deref(), Some("Ana"));

        let system_transcript_id: String = sqlx::query_scalar(
            "SELECT id FROM transcripts WHERE meeting_id = ? AND transcript = 'fala do sistema'",
        )
        .bind(&meeting_id)
        .fetch_one(&pool)
        .await
        .expect("system transcript must exist");
        SpeakersRepository::reassign_transcript(&pool, &meeting_id, &system_transcript_id, Some(8))
            .await
            .expect("speaker reassignment must succeed");
        let reassigned: Option<i64> =
            sqlx::query_scalar("SELECT speaker_id FROM transcripts WHERE id = ?")
                .bind(&system_transcript_id)
                .fetch_one(&pool)
                .await
                .expect("reassigned transcript must load");
        assert_eq!(reassigned, Some(8));
    }

    #[tokio::test]
    async fn source_migration_upgrades_populated_legacy_database() {
        let pool = in_memory_pool().await;
        let migrator = sqlx::migrate!("./migrations");

        for migration in migrator.iter() {
            if migration.version >= SOURCE_MIGRATION_VERSION {
                continue;
            }
            sqlx::raw_sql(migration.sql.as_ref())
                .execute(&pool)
                .await
                .unwrap_or_else(|e| panic!("legacy migration {} failed: {}", migration.version, e));
        }

        sqlx::query(
            "INSERT INTO meetings (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        )
        .bind("meeting-legacy")
        .bind("Reunião antiga")
        .bind("2025-01-01T00:00:00Z")
        .bind("2025-01-01T00:00:00Z")
        .execute(&pool)
        .await
        .expect("insert legacy meeting");
        sqlx::query(
            "INSERT INTO transcripts (id, meeting_id, transcript, timestamp) VALUES (?, ?, ?, ?)",
        )
        .bind("transcript-legacy")
        .bind("meeting-legacy")
        .bind("gravado antes da migração")
        .bind("2025-01-01T00:00:00Z")
        .execute(&pool)
        .await
        .expect("insert legacy transcript row");

        let source_migration = migrator
            .iter()
            .find(|m| m.version == SOURCE_MIGRATION_VERSION)
            .expect("add_transcript_source migration must exist");
        sqlx::raw_sql(source_migration.sql.as_ref())
            .execute(&pool)
            .await
            .expect("source migration must apply to a populated legacy database");

        let speaker_migration = migrator
            .iter()
            .find(|m| m.version == SPEAKER_MIGRATION_VERSION)
            .expect("speaker assignments migration must exist");
        sqlx::raw_sql(speaker_migration.sql.as_ref())
            .execute(&pool)
            .await
            .expect("speaker migration must apply to a populated legacy database");

        let legacy: Transcript = sqlx::query_as("SELECT * FROM transcripts WHERE id = ?")
            .bind("transcript-legacy")
            .fetch_one(&pool)
            .await
            .expect("legacy row must still load after migration");
        assert_eq!(legacy.source, None, "pre-migration rows must load with NULL source");
        assert_eq!(
            legacy.speaker_id, None,
            "pre-migration rows must load with NULL speaker"
        );
        assert_eq!(legacy.transcript, "gravado antes da migração");
    }
}
