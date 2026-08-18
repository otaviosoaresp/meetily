//! Real-time, system-channel-only speaker clustering.
//!
//! The audio pipeline already provides complete 16 kHz speech segments from a
//! stateful VAD. This module turns each completed system segment into a
//! speaker embedding and matches it against meeting-local clusters. It owns
//! model loading and clustering state so the hot capture callback never loads
//! a model or performs inference.

use anyhow::{anyhow, Result};
use futures_util::StreamExt;
use log::{info, warn};
use pyannote_rs::{EmbeddingExtractor, EmbeddingManager};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tauri::{AppHandle, Emitter, Manager, Runtime};
use tokio::io::AsyncWriteExt;

pub const EMBEDDING_MODEL_FILE: &str = "wespeaker_en_voxceleb_CAM++.onnx";
pub const EMBEDDING_MODEL_URL: &str = "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM%2B%2B.onnx";

const DEFAULT_MAX_SPEAKERS: usize = 16;
const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.5;
const MIN_EMBEDDING_SAMPLES: usize = 16_000;
/// Keep each embedding focused on one conversational turn. A VAD segment can
/// be long when multiple remote speakers talk without a pause, so it must not
/// be represented by one mixed embedding.
const EMBEDDING_WINDOW_SAMPLES: usize = 16_000 * 4;
/// Bound live inference cost while still sampling enough of a long segment to
/// determine whether one speaker is dominant.
const MAX_EMBEDDING_WINDOWS: usize = 4;
const DOWNLOAD_PROGRESS_EVENT: &str = "diarization-model-download-progress";
const FALLBACK_MODEL_TOTAL_MB: f64 = 28.0;

static DOWNLOAD_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// The persisted/displayed speaker ID is zero-based. pyannote-rs IDs are
/// one-based, so the conversion is kept at this boundary.
pub struct StreamingSpeakerDiarizer {
    extractor: EmbeddingExtractor,
    speakers: EmbeddingManager,
    similarity_threshold: f32,
}

impl StreamingSpeakerDiarizer {
    pub fn new(model_path: &Path) -> Result<Self> {
        if !model_path.is_file() {
            return Err(anyhow!(
                "speaker embedding model not found: {}",
                model_path.display()
            ));
        }

        let extractor = EmbeddingExtractor::new(model_path)
            .map_err(|error| anyhow!("failed to load speaker embedding model: {error:?}"))?;

        info!(
            "Streaming speaker diarizer loaded from {}",
            model_path.display()
        );
        Ok(Self {
            extractor,
            speakers: EmbeddingManager::new(DEFAULT_MAX_SPEAKERS),
            similarity_threshold: DEFAULT_SIMILARITY_THRESHOLD,
        })
    }

    /// Assign a stable meeting-local cluster to one completed VAD segment.
    /// Short segments are deliberately left unlabeled because their embedding
    /// is not reliable enough to create or move a cluster.
    pub fn identify(&mut self, samples: &[f32], sample_rate: u32) -> Option<u32> {
        let samples_16k = if sample_rate == 16_000 {
            samples.to_vec()
        } else {
            crate::audio::audio_processing::resample_audio(samples, sample_rate, 16_000)
        };

        if samples_16k.len() < MIN_EMBEDDING_SAMPLES {
            return None;
        }

        let assignments: Vec<u32> = samples_16k
            .chunks(EMBEDDING_WINDOW_SAMPLES)
            .take(MAX_EMBEDDING_WINDOWS)
            .filter_map(|window| {
                if window.len() < MIN_EMBEDDING_SAMPLES {
                    return None;
                }
                self.identify_window(window)
            })
            .collect();

        dominant_speaker(&assignments)
    }

    fn identify_window(&mut self, samples_16k: &[f32]) -> Option<u32> {
        let pcm: Vec<i16> = samples_16k
            .iter()
            .map(|sample| (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();

        let embedding = match self.extractor.compute(&pcm) {
            Ok(values) => values.collect::<Vec<f32>>(),
            Err(error) => {
                warn!("Speaker embedding failed; leaving segment as Outros: {error:?}");
                return None;
            }
        };

        let speaker_id = self
            .speakers
            .search_speaker(embedding.clone(), self.similarity_threshold)
            .or_else(|| self.speakers.search_speaker(embedding, 0.0));

        speaker_id.map(|id| id.saturating_sub(1) as u32)
    }
}

fn dominant_speaker(assignments: &[u32]) -> Option<u32> {
    let mut counts = BTreeMap::<u32, usize>::new();
    for speaker_id in assignments {
        *counts.entry(*speaker_id).or_default() += 1;
    }

    let (speaker_id, count) = counts.into_iter().max_by_key(|(_, count)| *count)?;
    (count * 2 > assignments.len()).then_some(speaker_id)
}

/// Resolve the per-user model location used by the Tauri commands and the
/// recording startup path. A caller may override it for development/tests.
pub fn embedding_model_path(models_dir: &Path) -> PathBuf {
    models_dir.join("diarization").join(EMBEDDING_MODEL_FILE)
}

fn app_model_path<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|error| format!("failed to resolve Meetily app data directory: {error}"))?;
    Ok(embedding_model_path(&app_data_dir.join("models")))
}

#[tauri::command]
pub fn diarization_get_model_status<R: Runtime>(
    app: AppHandle<R>,
) -> Result<DiarizationModelStatus, String> {
    let model_path = app_model_path(&app)?;
    Ok(DiarizationModelStatus {
        available: model_path.is_file(),
        model_path: model_path.to_string_lossy().into_owned(),
    })
}

#[derive(Debug, Clone, serde::Serialize)]
struct DownloadProgressPayload {
    model: String,
    progress: f64,
    downloaded_mb: f64,
    total_mb: f64,
    speed_mbps: f64,
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

fn emit_download_progress<R: Runtime>(
    app: &AppHandle<R>,
    progress: f64,
    downloaded_mb: f64,
    total_mb: f64,
    speed_mbps: f64,
    status: &str,
    error: Option<String>,
) {
    let _ = app.emit(
        DOWNLOAD_PROGRESS_EVENT,
        DownloadProgressPayload {
            model: EMBEDDING_MODEL_FILE.to_string(),
            progress,
            downloaded_mb,
            total_mb,
            speed_mbps,
            status: status.to_string(),
            error,
        },
    );
}

async fn run_model_download<R: Runtime>(
    app: &AppHandle<R>,
    model_path: &Path,
) -> Result<f64, String> {
    let model_dir = model_path
        .parent()
        .ok_or_else(|| "speaker model path has no parent directory".to_string())?;
    tokio::fs::create_dir_all(model_dir)
        .await
        .map_err(|error| format!("failed to create speaker model directory: {error}"))?;

    let temporary_path = model_path.with_extension("onnx.part");
    let response = reqwest::Client::new()
        .get(EMBEDDING_MODEL_URL)
        .send()
        .await
        .map_err(|error| format!("failed to download speaker model: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "speaker model download failed with HTTP status {}",
            response.status()
        ));
    }

    let total_mb = response
        .content_length()
        .map(|bytes| bytes as f64 / 1_048_576.0)
        .unwrap_or(FALLBACK_MODEL_TOTAL_MB);

    let mut output = tokio::fs::File::create(&temporary_path)
        .await
        .map_err(|error| format!("failed to create temporary speaker model: {error}"))?;
    let mut stream = response.bytes_stream();
    let started_at = Instant::now();
    let mut last_progress_emit = Instant::now();
    let mut downloaded_bytes: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk =
            chunk.map_err(|error| format!("speaker model download interrupted: {error}"))?;
        output
            .write_all(&chunk)
            .await
            .map_err(|error| format!("failed to write speaker model: {error}"))?;
        downloaded_bytes += chunk.len() as u64;

        if last_progress_emit.elapsed().as_millis() >= 250 {
            last_progress_emit = Instant::now();
            let downloaded_mb = downloaded_bytes as f64 / 1_048_576.0;
            let progress = (downloaded_mb / total_mb * 100.0).min(99.0);
            let elapsed_seconds = started_at.elapsed().as_secs_f64().max(0.001);
            emit_download_progress(
                app,
                progress,
                downloaded_mb,
                total_mb,
                downloaded_mb / elapsed_seconds,
                "downloading",
                None,
            );
        }
    }
    output
        .flush()
        .await
        .map_err(|error| format!("failed to flush speaker model: {error}"))?;
    drop(output);

    tokio::fs::rename(&temporary_path, &model_path)
        .await
        .map_err(|error| format!("failed to install speaker model: {error}"))?;

    Ok(downloaded_bytes as f64 / 1_048_576.0)
}

/// Download the CPU-safe embedding model into the per-user app-data directory.
/// The temporary file is kept beside the destination so a failed download can
/// never leave a partially written model that the recorder tries to load.
/// Progress, completion, and failure are emitted as toast-visible events; only
/// one download runs at a time.
async fn download_model_with_progress<R: Runtime>(
    app: &AppHandle<R>,
    model_path: &Path,
) -> Result<(), String> {
    if DOWNLOAD_IN_PROGRESS.swap(true, Ordering::SeqCst) {
        return Err("speaker model download already in progress".to_string());
    }

    let result = if model_path.is_file() {
        Ok(0.0)
    } else {
        let download_result = run_model_download(app, model_path).await;
        match &download_result {
            Ok(downloaded_mb) => emit_download_progress(
                app,
                100.0,
                *downloaded_mb,
                *downloaded_mb,
                0.0,
                "completed",
                None,
            ),
            Err(error) => {
                emit_download_progress(app, 0.0, 0.0, 0.0, 0.0, "error", Some(error.clone()))
            }
        }
        download_result
    };

    DOWNLOAD_IN_PROGRESS.store(false, Ordering::SeqCst);
    result.map(|_| ())
}

/// Fire-and-forget provisioning used by every recording start: recording never
/// waits for the download, system lines stay plain "Outros" until the model is
/// installed, and a failed download leaves the recording untouched.
pub fn ensure_model_available<R: Runtime>(app: &AppHandle<R>) {
    let model_path = match app_model_path(app) {
        Ok(path) => path,
        Err(error) => {
            warn!("Cannot resolve speaker model path; system lines stay plain Outros: {error}");
            return;
        }
    };
    if model_path.is_file() || DOWNLOAD_IN_PROGRESS.load(Ordering::SeqCst) {
        return;
    }

    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        match download_model_with_progress(&app, &model_path).await {
            Ok(()) => info!(
                "Speaker embedding model installed at {}; the next recording will label system speakers",
                model_path.display()
            ),
            Err(error) => warn!(
                "Speaker model download failed; system lines stay plain Outros: {error}"
            ),
        }
    });
}

#[tauri::command]
pub async fn diarization_download_model<R: Runtime>(
    app: AppHandle<R>,
) -> Result<DiarizationModelStatus, String> {
    let model_path = app_model_path(&app)?;
    if !model_path.is_file() {
        download_model_with_progress(&app, &model_path).await?;
    }

    Ok(DiarizationModelStatus {
        available: true,
        model_path: model_path.to_string_lossy().into_owned(),
    })
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DiarizationModelStatus {
    pub model_path: String,
    pub available: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_path_is_scoped_to_meetily_models() {
        let path = embedding_model_path(Path::new("/tmp/meetily/models"));
        assert!(path.ends_with("models/diarization/wespeaker_en_voxceleb_CAM++.onnx"));
    }

    fn read_mono_16k_wav(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("sample wav must exist");
        let data_offset = bytes
            .windows(4)
            .position(|window| window == b"data")
            .expect("wav must contain a data chunk")
            + 8;
        bytes[data_offset..]
            .chunks_exact(2)
            .map(|pair| i16::from_le_bytes([pair[0], pair[1]]) as f32 / i16::MAX as f32)
            .collect()
    }

    /// End-to-end check of the streaming path on the real CPU model: multi-
    /// speaker speech must split into more than one meeting-local cluster, and
    /// re-identifying earlier audio must return the cluster it was assigned
    /// live (the labels users see as "Outros 1", "Outros 2", ...).
    ///
    /// Ignored by default: set MEETILY_DIARIZATION_TEST_ASSETS to a directory
    /// containing `wespeaker_en_voxceleb_CAM++.onnx` and `6_speakers.wav`
    /// (both published at github.com/thewh1teagle/pyannote-rs releases) and
    /// run with `cargo test -- --ignored`.
    #[test]
    #[ignore = "needs MEETILY_DIARIZATION_TEST_ASSETS with the ONNX model and sample wav"]
    fn real_model_assigns_stable_incremental_clusters() {
        let Ok(assets) = std::env::var("MEETILY_DIARIZATION_TEST_ASSETS") else {
            eprintln!("MEETILY_DIARIZATION_TEST_ASSETS not set; nothing to do");
            return;
        };
        let assets = PathBuf::from(assets);
        let samples = read_mono_16k_wav(&assets.join("6_speakers.wav"));
        let mut diarizer = StreamingSpeakerDiarizer::new(&assets.join(EMBEDDING_MODEL_FILE))
            .expect("embedding model must load on CPU");

        let window = 4 * 16_000;
        let mut assignments = Vec::new();
        for (index, chunk) in samples.chunks(window).enumerate() {
            let speaker = diarizer.identify(chunk, 16_000);
            let start = index * 4;
            match speaker {
                Some(id) => eprintln!(
                    "segment {index:>2} ({start:>2}s-{:>2}s) -> Outros {}",
                    start + 4,
                    id + 1
                ),
                None => eprintln!("segment {index:>2} ({start:>2}s-...) -> unlabeled (Outros)"),
            }
            assignments.push((index, chunk, speaker));
        }

        let clusters: std::collections::BTreeSet<u32> = assignments
            .iter()
            .filter_map(|(_, _, speaker)| *speaker)
            .collect();
        eprintln!("distinct clusters: {clusters:?}");
        assert!(
            clusters.len() >= 2,
            "multi-speaker speech must produce at least two clusters, got {clusters:?}"
        );

        for (index, chunk, speaker) in &assignments {
            if let Some(expected) = speaker {
                let again = diarizer.identify(chunk, 16_000);
                assert_eq!(
                    again,
                    Some(*expected),
                    "segment {index} must stay in its live-assigned cluster"
                );
            }
        }
    }

    /// A long VAD segment may contain several remote speakers. It must remain
    /// unlabeled when no speaker has a strict majority instead of inheriting a
    /// cluster from the first few seconds of mixed audio.
    #[test]
    #[ignore = "needs MEETILY_DIARIZATION_TEST_ASSETS with the ONNX model and sample wav"]
    fn real_model_does_not_collapse_mixed_long_segments() {
        let Ok(assets) = std::env::var("MEETILY_DIARIZATION_TEST_ASSETS") else {
            eprintln!("MEETILY_DIARIZATION_TEST_ASSETS not set; nothing to do");
            return;
        };
        let assets = PathBuf::from(assets);
        let samples = read_mono_16k_wav(&assets.join("6_speakers.wav"));
        let mut diarizer = StreamingSpeakerDiarizer::new(&assets.join(EMBEDDING_MODEL_FILE))
            .expect("embedding model must load on CPU");

        let assignments: Vec<Option<u32>> = samples
            .chunks(16 * 16_000)
            .map(|chunk| diarizer.identify(chunk, 16_000))
            .collect();
        assert!(
            assignments.iter().any(Option::is_none),
            "mixed long segments must stay plain Outros when no speaker dominates: {assignments:?}"
        );
    }
}
