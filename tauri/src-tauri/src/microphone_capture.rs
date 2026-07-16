//! Native microphone recording for dictation and voice samples.
//!
//! WKWebView's getUserMedia/MediaRecorder implementation rejects otherwise
//! valid audio constraints on some macOS builds. Capturing through cpal keeps
//! the hotkey path independent of the browser media stack and emits the WAV
//! payload the backend already accepts.

use base64::{engine::general_purpose, Engine as _};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use hound::{WavSpec, WavWriter};
use std::io::Cursor;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[cfg(target_os = "macos")]
fn request_macos_microphone_access() -> Result<(), String> {
    use block2::RcBlock;
    use objc::runtime::Object;
    use objc::{class, msg_send, sel, sel_impl};

    #[link(name = "AVFoundation", kind = "framework")]
    extern "C" {}

    // AVMediaTypeAudio is the CoreMedia type string "soun".
    let media_type: *mut Object = unsafe {
        msg_send![class!(NSString), stringWithUTF8String: b"soun\0".as_ptr()]
    };
    if media_type.is_null() {
        return Err("Could not initialize macOS microphone permission request.".to_string());
    }

    let status: isize = unsafe {
        msg_send![class!(AVCaptureDevice), authorizationStatusForMediaType: media_type]
    };
    match status {
        3 => Ok(()),
        1 => Err("Microphone access is restricted by macOS.".to_string()),
        2 => Err(
            "Microphone access is denied. Allow Voicebox in System Settings > Privacy & Security > Microphone."
                .to_string(),
        ),
        0 => {
            let (granted_tx, granted_rx) = std::sync::mpsc::channel();
            let completion: RcBlock<dyn Fn(i8)> = RcBlock::new(move |granted: i8| {
                let _ = granted_tx.send(granted != 0);
            });
            unsafe {
                let _: () = msg_send![
                    class!(AVCaptureDevice),
                    requestAccessForMediaType: media_type
                    completionHandler: &*completion
                ];
            }
            match granted_rx.recv_timeout(Duration::from_secs(15)) {
                Ok(true) => Ok(()),
                Ok(false) => Err(
                    "Microphone access was denied. Allow Voicebox in System Settings > Privacy & Security > Microphone."
                        .to_string(),
                ),
                Err(_) => Err("Timed out waiting for macOS microphone permission.".to_string()),
            }
        }
        _ => Err("macOS returned an unknown microphone permission state.".to_string()),
    }
}

#[cfg(not(target_os = "macos"))]
fn request_macos_microphone_access() -> Result<(), String> {
    Ok(())
}

pub struct MicrophoneCaptureState {
    samples: Arc<Mutex<Vec<f32>>>,
    chunk_cursor: Arc<Mutex<usize>>,
    sample_rate: Arc<Mutex<u32>>,
    channels: Arc<Mutex<u16>>,
    error: Arc<Mutex<Option<String>>>,
    stop_tx: Arc<Mutex<Option<mpsc::Sender<()>>>>,
}

impl MicrophoneCaptureState {
    pub fn new() -> Self {
        Self {
            samples: Arc::new(Mutex::new(Vec::new())),
            chunk_cursor: Arc::new(Mutex::new(0)),
            sample_rate: Arc::new(Mutex::new(44_100)),
            channels: Arc::new(Mutex::new(1)),
            error: Arc::new(Mutex::new(None)),
            stop_tx: Arc::new(Mutex::new(None)),
        }
    }

    fn reset(&self) {
        if let Some(tx) = self.stop_tx.lock().unwrap().take() {
            let _ = tx.send(());
        }
        self.samples.lock().unwrap().clear();
        *self.chunk_cursor.lock().unwrap() = 0;
        *self.error.lock().unwrap() = None;
    }
}

pub async fn start_capture(
    state: &MicrophoneCaptureState,
    max_duration_secs: u32,
) -> Result<(), String> {
    state.reset();

    request_macos_microphone_access()?;

    let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), String>>(1);
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    *state.stop_tx.lock().unwrap() = Some(stop_tx);

    let samples = state.samples.clone();
    let sample_rate_state = state.sample_rate.clone();
    let channels_state = state.channels.clone();
    let error = state.error.clone();

    // cpal's CoreAudio stream is intentionally !Send/!Sync. Keep the entire
    // device and stream lifecycle on one dedicated thread and expose only
    // Send/Sync sample buffers through Tauri state.
    thread::spawn(move || {
        let host = cpal::default_host();
        let mut candidates = Vec::new();
        if let Some(default_device) = host.default_input_device() {
            candidates.push(default_device);
        }
        if let Ok(input_devices) = host.input_devices() {
            candidates.extend(input_devices);
        }

        let selected_device = candidates.into_iter().find_map(|candidate| {
            match candidate.default_input_config() {
                Ok(config) => Some((candidate, config)),
                Err(err) => {
                    eprintln!("Ignoring CoreAudio device without a usable input stream: {err}");
                    None
                }
            }
        });
        let (device, supported_config) = match selected_device {
            Some(selected) => selected,
            None => {
                let _ = ready_tx.send(Err(
                    "No microphone input device is available in macOS. Connect a microphone and select it in System Settings > Sound > Input."
                        .to_string(),
                ));
                return;
            }
        };
        let device_name = device.name().unwrap_or_else(|_| "default microphone".to_string());
        let sample_format = supported_config.sample_format();
        let config: StreamConfig = supported_config.config();
        let sample_rate = config.sample_rate.0;
        let channels = config.channels;
        *sample_rate_state.lock().unwrap() = sample_rate;
        *channels_state.lock().unwrap() = channels;

        let error_for_callback = error.clone();
        let err_fn = move |err: cpal::StreamError| {
            let message = format!("Microphone stream error: {err}");
            eprintln!("{message}");
            *error_for_callback.lock().unwrap() = Some(message);
        };

        let stream = match sample_format {
            SampleFormat::F32 => {
                let samples = samples.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[f32], _| samples.lock().unwrap().extend_from_slice(data),
                    err_fn,
                    None,
                )
            }
            SampleFormat::I16 => {
                let samples = samples.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[i16], _| {
                        let mut output = samples.lock().unwrap();
                        output.extend(data.iter().map(|sample| *sample as f32 / 32_768.0));
                    },
                    err_fn,
                    None,
                )
            }
            SampleFormat::U16 => {
                let samples = samples.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[u16], _| {
                        let mut output = samples.lock().unwrap();
                        output.extend(data.iter().map(|sample| *sample as f32 / 32_768.0 - 1.0));
                    },
                    err_fn,
                    None,
                )
            }
            _format => Err(cpal::BuildStreamError::StreamConfigNotSupported),
        };

        let stream = match stream {
            Ok(stream) => stream,
            Err(err) => {
                let message = format!("Could not start microphone input: {err}");
                *error.lock().unwrap() = Some(message.clone());
                let _ = ready_tx.send(Err(message));
                return;
            }
        };

        if let Err(err) = stream.play() {
            let message = format!("Could not start microphone stream: {err}");
            *error.lock().unwrap() = Some(message.clone());
            let _ = ready_tx.send(Err(message));
            return;
        }

        eprintln!(
            "Native microphone capture started: {device_name}, {sample_rate}Hz, {channels} channel(s)"
        );
        if ready_tx.send(Ok(())).is_err() {
            return;
        }

        if max_duration_secs > 0 {
            let _ = stop_rx.recv_timeout(Duration::from_secs(max_duration_secs as u64));
        } else {
            let _ = stop_rx.recv();
        }
        drop(stream);
        eprintln!("Native microphone capture stopped");
    });

    ready_rx
        .recv_timeout(Duration::from_secs(5))
        .map_err(|_| "Timed out while starting microphone capture.".to_string())?
}

pub async fn stop_capture(state: &MicrophoneCaptureState) -> Result<String, String> {
    if let Some(tx) = state.stop_tx.lock().unwrap().take() {
        let _ = tx.send(());
    }
    tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;

    if let Some(error) = state.error.lock().unwrap().clone() {
        return Err(error);
    }

    let samples = state.samples.lock().unwrap().clone();
    if samples.is_empty() {
        return Err("No microphone samples captured. Check microphone permission and input device.".to_string());
    }

    let sample_rate = *state.sample_rate.lock().unwrap();
    let channels = *state.channels.lock().unwrap();
    let wav_data = samples_to_wav(&samples, sample_rate, channels)?;
    Ok(general_purpose::STANDARD.encode(wav_data))
}

pub async fn read_chunk(
    state: &MicrophoneCaptureState,
    min_duration_ms: u32,
) -> Result<Option<String>, String> {
    if let Some(error) = state.error.lock().unwrap().clone() {
        return Err(error);
    }

    let sample_rate = *state.sample_rate.lock().unwrap();
    let channels = *state.channels.lock().unwrap();
    let min_samples = ((sample_rate as u64)
        .saturating_mul(channels as u64)
        .saturating_mul(min_duration_ms as u64)
        / 1000) as usize;

    let chunk = {
        let samples = state.samples.lock().unwrap();
        let mut cursor = state.chunk_cursor.lock().unwrap();
        if samples.len().saturating_sub(*cursor) < min_samples {
            return Ok(None);
        }
        let chunk = samples[*cursor..].to_vec();
        *cursor = samples.len();
        chunk
    };

    if chunk.is_empty() {
        return Ok(None);
    }

    let wav_data = samples_to_wav(&chunk, sample_rate, channels)?;
    Ok(Some(general_purpose::STANDARD.encode(wav_data)))
}

pub fn is_supported() -> bool {
    cpal::default_host().default_input_device().is_some()
}

fn samples_to_wav(samples: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<u8>, String> {
    let mut buffer = Vec::new();
    let cursor = Cursor::new(&mut buffer);
    let spec = WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = WavWriter::new(cursor, spec)
        .map_err(|e| format!("Could not create microphone WAV: {e}"))?;

    for sample in samples {
        let sample = sample.clamp(-1.0, 1.0);
        writer
            .write_sample((sample * 32_767.0) as i16)
            .map_err(|e| format!("Could not write microphone WAV: {e}"))?;
    }
    writer
        .finalize()
        .map_err(|e| format!("Could not finalize microphone WAV: {e}"))?;
    Ok(buffer)
}
