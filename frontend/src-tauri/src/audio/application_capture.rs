use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// The system-audio source used for one recording.
///
/// Global capture is intentionally the default. Application capture is an
/// explicit per-meeting choice and never falls back to the sink monitor.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct AudioCaptureSelection {
    pub mode: AudioCaptureMode,
    pub object_serial: Option<u64>,
    pub application_name: Option<String>,
    pub media_name: Option<String>,
    pub process_name: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AudioCaptureMode {
    #[default]
    Global,
    Application,
}

impl AudioCaptureSelection {
    pub fn global() -> Self {
        Self::default()
    }

    pub fn is_application(&self) -> bool {
        self.mode == AudioCaptureMode::Application
    }

    pub fn validate(&self) -> Result<()> {
        if !self.is_application() {
            return Ok(());
        }

        if self.object_serial.is_none() {
            return Err(anyhow!(
                "No application audio stream was selected. Choose an application/media stream or switch to global system audio."
            ));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ApplicationAudioStream {
    pub object_serial: u64,
    pub node_id: u32,
    pub application_name: String,
    pub media_name: String,
    pub process_name: String,
    pub node_name: String,
}

#[cfg(target_os = "linux")]
mod linux {
    use super::{ApplicationAudioStream, AudioCaptureSelection};
    use crate::audio::pipeline::AudioCapture;
    use crate::audio::recording_state::{AudioError, DeviceType, RecordingState};
    use anyhow::{anyhow, Context, Result};
    use log::{info, warn};
    use pipewire as pw;
    use pw::properties::properties;
    use pw::registry::GlobalObject;
    use pw::spa::param::audio::{AudioFormat, AudioInfoRaw};
    use pw::spa::pod::Pod;
    use pw::spa::utils::Direction;
    use std::io::Cursor;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc as std_mpsc, Arc, Mutex};
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    const NODE_INTERFACE: &str = "PipeWire:Interface:Node";
    const MEDIA_CLASS: &str = "media.class";
    const APPLICATION_NAME: &str = "application.name";
    const MEDIA_NAME: &str = "media.name";
    const PROCESS_BINARY: &str = "application.process.binary";
    const PROCESS_ID: &str = "application.process.id";
    const NODE_NAME: &str = "node.name";
    const OBJECT_SERIAL: &str = "object.serial";

    pub fn list_application_audio_streams() -> Result<Vec<ApplicationAudioStream>> {
        run_registry_query(|streams| streams.collect::<Vec<_>>())
    }

    /// Resolve the saved serial first. If PipeWire recreated the node, use the
    /// saved identity to reacquire exactly one equivalent stream. Ambiguity is
    /// an error because selected mode must never capture a different app.
    fn resolve_target(selection: &AudioCaptureSelection) -> Result<ApplicationAudioStream> {
        let streams = list_application_audio_streams()?;

        if let Some(serial) = selection.object_serial {
            if let Some(stream) = streams.iter().find(|stream| stream.object_serial == serial) {
                return Ok(stream.clone());
            }
        }

        let matches: Vec<ApplicationAudioStream> = streams
            .into_iter()
            .filter(|stream| {
                selection
                    .application_name
                    .as_deref()
                    .map(|value| value == stream.application_name)
                    .unwrap_or(false)
                    && selection
                        .media_name
                        .as_deref()
                        .map(|value| value == stream.media_name)
                        .unwrap_or(false)
                    && selection
                        .process_name
                        .as_deref()
                        .map(|value| value == stream.process_name)
                        .unwrap_or(false)
            })
            .collect();

        match matches.as_slice() {
            [stream] => {
                info!(
                    "Reacquired selected PipeWire stream by identity: serial {} -> {}",
                    selection.object_serial.unwrap_or_default(),
                    stream.object_serial
                );
                Ok(stream.clone())
            }
            [] => Err(anyhow!(
                "The selected application/media stream is unavailable. Switch to global system audio or choose another stream."
            )),
            _ => Err(anyhow!(
                "The selected application exposes multiple matching media streams. Refresh the picker and choose one stream explicitly."
            )),
        }
    }

    pub struct PipeWireAudioStream {
        stop_sender: Option<pw::channel::Sender<()>>,
        thread: Option<JoinHandle<()>>,
    }

    // PipeWire's callback owns the stream on its dedicated thread. The handle
    // and channel are the only values shared with the recording thread.
    unsafe impl Send for PipeWireAudioStream {}

    impl PipeWireAudioStream {
        pub fn start(
            selection: &AudioCaptureSelection,
            state: Arc<RecordingState>,
        ) -> Result<Self> {
            selection.validate()?;
            let target = resolve_target(selection)?;
            let device = Arc::new(crate::audio::devices::AudioDevice::new(
                format!(
                    "{} — {} ({})",
                    target.application_name, target.media_name, target.process_name
                ),
                crate::audio::devices::DeviceType::Output,
            ));
            let capture =
                AudioCapture::new(device, state, 48000, 2, DeviceType::System, None);

            let (ready_sender, ready_receiver) = std_mpsc::channel::<Result<()>>();
            let (stop_sender, stop_receiver) = pw::channel::channel::<()>();
            let target_serial = target.object_serial;

            let thread = thread::Builder::new()
                .name("meetily-pipewire-capture".to_string())
                .spawn(move || {
                    if let Err(error) =
                        run_capture_thread(target_serial, capture, stop_receiver, ready_sender)
                    {
                        warn!("PipeWire capture thread exited with error: {}", error);
                    }
                })
                .context("Failed to start PipeWire capture thread")?;

            match ready_receiver.recv_timeout(Duration::from_secs(5)) {
                Ok(Ok(())) => Ok(Self {
                    stop_sender: Some(stop_sender),
                    thread: Some(thread),
                }),
                Ok(Err(error)) => {
                    let _ = stop_sender.send(());
                    let _ = thread.join();
                    Err(error)
                }
                Err(error) => {
                    let _ = stop_sender.send(());
                    let _ = thread.join();
                    Err(anyhow!(
                        "Timed out while connecting to the selected application audio stream: {}",
                        error
                    ))
                }
            }
        }

        pub fn stop(mut self) -> Result<()> {
            if let Some(sender) = self.stop_sender.take() {
                let _ = sender.send(());
            }
            if let Some(thread) = self.thread.take() {
                thread
                    .join()
                    .map_err(|_| anyhow!("PipeWire capture thread panicked"))?;
            }
            Ok(())
        }
    }

    fn run_capture_thread(
        target_serial: u64,
        capture: AudioCapture,
        stop_receiver: pw::channel::Receiver<()>,
        ready_sender: std_mpsc::Sender<Result<()>>,
    ) -> Result<()> {
        pw::init();
        let main_loop = pw::main_loop::MainLoopRc::new(None)?;
        let context = pw::context::ContextRc::new(&main_loop, None)?;
        let core = context.connect_rc(None)?;

        let mut props = properties! {
            *pw::keys::MEDIA_TYPE => "Audio",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Music",
            *pw::keys::NODE_DONT_RECONNECT => "true",
            "node.dont-fallback" => "true",
        };
        props.insert(*pw::keys::TARGET_OBJECT, target_serial.to_string());

        let stream = pw::stream::StreamBox::new(&core, "meetily-application-audio", props)?;
        let mut format = AudioInfoRaw::new();
        format.set_format(AudioFormat::F32LE);
        format.set_rate(48000);
        format.set_channels(2);
        let format_for_connection = format.clone();
        let pod_bytes = serialize_audio_format(format_for_connection)?;
        let pod = Pod::from_bytes(&pod_bytes)
            .ok_or_else(|| anyhow!("Failed to build PipeWire audio format pod"))?;
        let mut params = [pod];

        let reported_error = Arc::new(AtomicBool::new(false));
        let error_state = reported_error.clone();
        let main_loop_for_error = main_loop.clone();
        let user_data = CaptureUserData {
            capture,
            format,
        };
        let listener = stream
            .add_local_listener_with_user_data(user_data)
            .state_changed(move |_, data, _, new_state| {
                if let pw::stream::StreamState::Error(message) = new_state {
                    if !error_state.swap(true, Ordering::SeqCst) {
                        data.capture.state().report_error(AudioError::SelectedAudioUnavailable(
                            format!(
                                "Selected application audio stream disappeared: {}. Switch to global system audio.",
                                message
                            ),
                        ));
                    }
                    main_loop_for_error.quit();
                }
            })
            .param_changed(|_, data, id, param| {
                if id == pw::spa::param::ParamType::Format.as_raw() {
                    if let Some(param) = param {
                        if data.format.parse(param).is_err() {
                            warn!("PipeWire selected audio format could not be parsed");
                        }
                    }
                }
            })
            .process(|stream, data| {
                let Some(mut buffer) = stream.dequeue_buffer() else {
                    return;
                };
                for raw_data in buffer.datas_mut() {
                    let chunk = raw_data.chunk();
                    let offset = chunk.offset() as usize;
                    let end = offset.saturating_add(chunk.size() as usize);
                    let Some(bytes) = raw_data.data() else {
                        continue;
                    };
                    if end > bytes.len() || offset >= end {
                        continue;
                    }
                    let samples: Vec<f32> = bytes[offset..end]
                        .chunks_exact(std::mem::size_of::<f32>())
                        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                        .collect();
                    if !samples.is_empty() {
                        data.capture.process_audio_data(&samples);
                    }
                }
            })
            .register()?;

        stream.connect(
            Direction::Input,
            None,
            pw::stream::StreamFlags::AUTOCONNECT
                | pw::stream::StreamFlags::MAP_BUFFERS
                | pw::stream::StreamFlags::RT_PROCESS,
            &mut params,
        )?;

        ready_sender
            .send(Ok(()))
            .map_err(|_| anyhow!("PipeWire capture startup receiver closed"))?;

        let _stop_receiver = stop_receiver.attach(main_loop.loop_(), {
            let main_loop = main_loop.clone();
            move |_| main_loop.quit()
        });

        main_loop.run();
        drop(listener);
        Ok(())
    }

    struct CaptureUserData {
        capture: AudioCapture,
        format: AudioInfoRaw,
    }

    fn serialize_audio_format(format: AudioInfoRaw) -> Result<Vec<u8>> {
        let object = pw::spa::pod::Object {
            type_: pw::spa::utils::SpaTypes::ObjectParamFormat.as_raw(),
            id: pw::spa::param::ParamType::EnumFormat.as_raw(),
            properties: format.into(),
        };
        Ok(pw::spa::pod::serialize::PodSerializer::serialize(
            Cursor::new(Vec::new()),
            &pw::spa::pod::Value::Object(object),
        )?
        .0
        .into_inner())
    }

    fn run_registry_query<T, F>(build_result: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(std::vec::IntoIter<ApplicationAudioStream>) -> T + Send + 'static,
    {
        let (result_sender, result_receiver) = std_mpsc::channel::<Result<T>>();
        thread::Builder::new()
            .name("meetily-pipewire-enumeration".to_string())
            .spawn(move || {
                let result = (|| -> Result<T> {
                    pw::init();
                    let main_loop = pw::main_loop::MainLoopRc::new(None)?;
                    let context = pw::context::ContextRc::new(&main_loop, None)?;
                    let core = context.connect_rc(None)?;
                    let registry = core.get_registry_rc()?;
                    let streams = Arc::new(Mutex::new(Vec::new()));
                    let streams_for_listener = streams.clone();
                    let listener = registry
                        .add_listener_local()
                        .global(move |global| {
                            if let Some(stream) = application_stream_from_global(global) {
                                streams_for_listener.lock().unwrap().push(stream);
                            }
                        })
                        .register();
                    let timer = main_loop.loop_().add_timer({
                        let main_loop = main_loop.clone();
                        move |_| main_loop.quit()
                    });
                    timer.update_timer(Some(Duration::from_millis(150)), None);
                    main_loop.run();
                    drop(listener);
                    let streams = Arc::try_unwrap(streams)
                        .map_err(|_| anyhow!("PipeWire enumeration state still in use"))?
                        .into_inner()
                        .map_err(|_| anyhow!("PipeWire enumeration state was poisoned"))?;
                    Ok(build_result(streams.into_iter()))
                })();
                let _ = result_sender.send(result);
            })?;

        result_receiver
            .recv_timeout(Duration::from_secs(3))
            .map_err(|error| {
                anyhow!(
                    "Timed out while enumerating PipeWire application streams: {}",
                    error
                )
            })?
    }

    fn application_stream_from_global(
        global: &GlobalObject<&pw::spa::utils::dict::DictRef>,
    ) -> Option<ApplicationAudioStream> {
        if global.type_.to_str() != NODE_INTERFACE {
            return None;
        }
        let props = global.props.as_ref()?;
        if props.get(MEDIA_CLASS)? != "Stream/Output/Audio" {
            return None;
        }
        let serial = props.get(OBJECT_SERIAL)?.parse::<u64>().ok()?;
        let node_name = props.get(NODE_NAME).unwrap_or("Unknown node").to_string();
        let application_name = props
            .get(APPLICATION_NAME)
            .unwrap_or(node_name.as_str())
            .to_string();
        let media_name = props
            .get(MEDIA_NAME)
            .unwrap_or("Audio playback")
            .to_string();
        let process_name = props
            .get(PROCESS_BINARY)
            .map(str::to_string)
            .or_else(|| props.get(PROCESS_ID).map(|pid| format!("process {}", pid)))
            .unwrap_or_else(|| "Unknown process".to_string());

        Some(ApplicationAudioStream {
            object_serial: serial,
            node_id: global.id,
            application_name,
            media_name,
            process_name,
            node_name,
        })
    }

    pub use list_application_audio_streams as list;
    pub use PipeWireAudioStream as Stream;
}

#[cfg(target_os = "linux")]
pub use linux::{list, Stream};

#[cfg(not(target_os = "linux"))]
pub fn list() -> Result<Vec<ApplicationAudioStream>> {
    Ok(Vec::new())
}

#[cfg(not(target_os = "linux"))]
pub struct Stream;
