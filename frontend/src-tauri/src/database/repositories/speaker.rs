use crate::database::models::SpeakerModel;
use chrono::Utc;
use sqlx::{Error as SqlxError, SqlitePool};

pub struct SpeakersRepository;

impl SpeakersRepository {
    pub async fn list(pool: &SqlitePool, meeting_id: &str) -> Result<Vec<SpeakerModel>, SqlxError> {
        sqlx::query_as::<_, SpeakerModel>(
            "SELECT meeting_id, speaker_id, name, created_at, updated_at
             FROM meeting_speakers WHERE meeting_id = ? ORDER BY speaker_id ASC",
        )
        .bind(meeting_id)
        .fetch_all(pool)
        .await
    }

    pub async fn rename(
        pool: &SqlitePool,
        meeting_id: &str,
        speaker_id: i64,
        name: Option<String>,
    ) -> Result<SpeakerModel, SqlxError> {
        let now = Utc::now().to_rfc3339();
        sqlx::query(
            "INSERT INTO meeting_speakers (meeting_id, speaker_id, name, created_at, updated_at)
             VALUES (?, ?, ?, ?, ?)
             ON CONFLICT(meeting_id, speaker_id) DO UPDATE SET name = excluded.name, updated_at = excluded.updated_at",
        )
        .bind(meeting_id)
        .bind(speaker_id)
        .bind(name.filter(|value| !value.trim().is_empty()))
        .bind(&now)
        .bind(&now)
        .execute(pool)
        .await?;

        sqlx::query_as::<_, SpeakerModel>(
            "SELECT meeting_id, speaker_id, name, created_at, updated_at
             FROM meeting_speakers WHERE meeting_id = ? AND speaker_id = ?",
        )
        .bind(meeting_id)
        .bind(speaker_id)
        .fetch_one(pool)
        .await
    }

    pub async fn reassign_transcript(
        pool: &SqlitePool,
        meeting_id: &str,
        transcript_id: &str,
        speaker_id: Option<i64>,
    ) -> Result<bool, SqlxError> {
        let mut transaction = pool.begin().await?;

        let source = sqlx::query_scalar::<_, Option<String>>(
            "SELECT source FROM transcripts WHERE id = ? AND meeting_id = ?",
        )
        .bind(transcript_id)
        .bind(meeting_id)
        .fetch_optional(&mut *transaction)
        .await?;

        if source.flatten().as_deref() != Some("Outros") {
            transaction.rollback().await?;
            return Ok(false);
        }

        sqlx::query("UPDATE transcripts SET speaker_id = ? WHERE id = ? AND meeting_id = ?")
            .bind(speaker_id)
            .bind(transcript_id)
            .bind(meeting_id)
            .execute(&mut *transaction)
            .await?;

        if let Some(id) = speaker_id {
            let now = Utc::now().to_rfc3339();
            sqlx::query(
                "INSERT OR IGNORE INTO meeting_speakers (meeting_id, speaker_id, name, created_at, updated_at)
                 VALUES (?, ?, NULL, ?, ?)",
            )
            .bind(meeting_id)
            .bind(id)
            .bind(&now)
            .bind(&now)
            .execute(&mut *transaction)
            .await?;
        }

        transaction.commit().await?;
        Ok(true)
    }
}
