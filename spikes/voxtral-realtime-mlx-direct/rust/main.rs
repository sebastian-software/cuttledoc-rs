use std::{
    env,
    ffi::{CStr, CString, c_char, c_void},
    fs, ptr,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

const OK: i32 = 0;
const INVALID_ARGUMENT: i32 = 1;
const CANCELLED: i32 = 3;
const BACKPRESSURE: i32 = 5;
const NEEDS_AUDIO: i32 = 6;
const DONE: i32 = 7;

const INPUT_CHUNK_SAMPLES: usize = 1_280;
const MAX_PENDING_SAMPLES: usize = 10_240;
const MAX_INGEST_SAMPLES_PER_STEP: usize = 5_120;
const SAMPLE_RATE_HZ: usize = 16_000;
const MAX_GENERATED_TOKENS: usize = 4_096;

unsafe extern "C" {
    fn cuttledoc_voxtral_mlx_inspect_model(
        model_directory: *const c_char,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_voxtral_mlx_probe_audio_frontend(
        model_directory: *const c_char,
        audio: *const f32,
        audio_len: usize,
        transcription_delay_ms: i32,
        device_kind: i32,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_voxtral_mlx_probe_causal_encoder(
        model_directory: *const c_char,
        audio: *const f32,
        audio_len: usize,
        transcription_delay_ms: i32,
        device_kind: i32,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_voxtral_mlx_transcribe(
        model_directory: *const c_char,
        audio: *const f32,
        audio_len: usize,
        transcription_delay_ms: i32,
        max_generated_tokens: usize,
        device_kind: i32,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_voxtral_mlx_session_create(
        model_directory: *const c_char,
        transcription_delay_ms: i32,
        max_generated_tokens: usize,
        max_decode_tokens_per_step: usize,
        device_kind: i32,
        max_pending_samples: usize,
        max_ingest_samples_per_step: usize,
        status_out: *mut i32,
        error_out: *mut *mut c_char,
    ) -> *mut c_void;
    fn cuttledoc_voxtral_mlx_session_feed(
        handle: *mut c_void,
        audio: *const f32,
        audio_len: usize,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_voxtral_mlx_session_close(handle: *mut c_void, error_out: *mut *mut c_char)
    -> i32;
    fn cuttledoc_voxtral_mlx_session_step(
        handle: *mut c_void,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_voxtral_mlx_session_cancel(handle: *mut c_void);
    fn cuttledoc_voxtral_mlx_session_destroy(handle: *mut c_void);
    fn cuttledoc_voxtral_mlx_free_string(value: *mut c_char);
}

struct Session {
    handle: *mut c_void,
}

impl Drop for Session {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cuttledoc_voxtral_mlx_session_destroy(self.handle) };
        }
    }
}

struct Response {
    status: i32,
    json: Option<String>,
    error: Option<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    match arguments.as_slice() {
        [command, model_directory] if command == "inspect" => inspect(model_directory),
        [command, model_directory, pcm_path, delay_ms, device] if command == "frontend" => {
            frontend(model_directory, pcm_path, delay_ms, device)
        }
        [command, model_directory, pcm_path, delay_ms, device] if command == "encoder" => {
            encoder(model_directory, pcm_path, delay_ms, device)
        }
        [
            command,
            model_directory,
            pcm_path,
            delay_ms,
            max_tokens,
            device,
        ] if command == "transcribe" => {
            transcribe(model_directory, pcm_path, delay_ms, max_tokens, device)
        }
        [command, model_directory, pcm_path, device] if command == "contract" => {
            contract(model_directory, pcm_path, device)
        }
        [
            command,
            model_directory,
            pcm_path,
            delay_ms,
            chunk_ms,
            max_decode_tokens,
            device,
        ] if command == "stream" => stream(
            model_directory,
            pcm_path,
            delay_ms,
            chunk_ms,
            max_decode_tokens,
            device,
        ),
        _ => Err(usage()),
    }
}

fn transcribe(
    model_directory: &str,
    pcm_path: &str,
    delay_ms: &str,
    max_tokens: &str,
    device: &str,
) -> Result<(), String> {
    let model_directory = c_string(model_directory, "model path")?;
    let audio = read_audio(pcm_path)?;
    let delay_ms = delay_ms
        .parse::<i32>()
        .map_err(|error| format!("delay must be a positive integer: {error}"))?;
    if delay_ms <= 0 {
        return Err("delay must be positive".to_owned());
    }
    let max_tokens = max_tokens
        .parse::<usize>()
        .map_err(|error| format!("max tokens must be a positive integer: {error}"))?;
    if max_tokens == 0 {
        return Err("max tokens must be positive".to_owned());
    }
    let device_kind = parse_device(device)?;
    let mut json = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        cuttledoc_voxtral_mlx_transcribe(
            model_directory.as_ptr(),
            audio.as_ptr(),
            audio.len(),
            delay_ms,
            max_tokens,
            device_kind,
            &mut json,
            &mut error,
        )
    };
    let response = Response {
        status,
        json: take_string(json),
        error: take_string(error),
    };
    if response.status != OK {
        return Err(response.error.unwrap_or_else(|| {
            format!("MLX transcription returned status {status} without a message")
        }));
    }
    println!(
        "{}",
        response
            .json
            .ok_or_else(|| "MLX transcription returned no JSON".to_owned())?
    );
    Ok(())
}

fn encoder(
    model_directory: &str,
    pcm_path: &str,
    delay_ms: &str,
    device: &str,
) -> Result<(), String> {
    let model_directory = c_string(model_directory, "model path")?;
    let audio = read_audio(pcm_path)?;
    let delay_ms = delay_ms
        .parse::<i32>()
        .map_err(|error| format!("delay must be a positive integer: {error}"))?;
    if delay_ms <= 0 {
        return Err("delay must be positive".to_owned());
    }
    let device_kind = parse_device(device)?;
    let mut json = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        cuttledoc_voxtral_mlx_probe_causal_encoder(
            model_directory.as_ptr(),
            audio.as_ptr(),
            audio.len(),
            delay_ms,
            device_kind,
            &mut json,
            &mut error,
        )
    };
    let response = Response {
        status,
        json: take_string(json),
        error: take_string(error),
    };
    if response.status != OK {
        return Err(response
            .error
            .unwrap_or_else(|| format!("MLX encoder returned status {status} without a message")));
    }
    println!(
        "{}",
        response
            .json
            .ok_or_else(|| "MLX encoder returned no JSON".to_owned())?
    );
    Ok(())
}

fn frontend(
    model_directory: &str,
    pcm_path: &str,
    delay_ms: &str,
    device: &str,
) -> Result<(), String> {
    let model_directory = c_string(model_directory, "model path")?;
    let audio = read_audio(pcm_path)?;
    let delay_ms = delay_ms
        .parse::<i32>()
        .map_err(|error| format!("delay must be a positive integer: {error}"))?;
    if delay_ms <= 0 {
        return Err("delay must be positive".to_owned());
    }
    let device_kind = parse_device(device)?;
    let mut json = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        cuttledoc_voxtral_mlx_probe_audio_frontend(
            model_directory.as_ptr(),
            audio.as_ptr(),
            audio.len(),
            delay_ms,
            device_kind,
            &mut json,
            &mut error,
        )
    };
    let response = Response {
        status,
        json: take_string(json),
        error: take_string(error),
    };
    if response.status != OK {
        return Err(response.error.unwrap_or_else(|| {
            format!("MLX frontend returned status {status} without a message")
        }));
    }
    println!(
        "{}",
        response
            .json
            .ok_or_else(|| "MLX frontend returned no JSON".to_owned())?
    );
    Ok(())
}

fn inspect(model_directory: &str) -> Result<(), String> {
    let model_directory = c_string(model_directory, "model path")?;
    let mut json = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        cuttledoc_voxtral_mlx_inspect_model(model_directory.as_ptr(), &mut json, &mut error)
    };
    let response = Response {
        status,
        json: take_string(json),
        error: take_string(error),
    };
    if response.status != OK {
        return Err(response.error.unwrap_or_else(|| {
            format!("MLX model inspection returned status {status} without a message")
        }));
    }
    println!(
        "{}",
        response
            .json
            .ok_or_else(|| "MLX model inspection returned no JSON".to_owned())?
    );
    Ok(())
}

fn contract(model_directory: &str, pcm_path: &str, device: &str) -> Result<(), String> {
    let model_directory = c_string(model_directory, "model path")?;
    let device_kind = parse_device(device)?;
    let audio = read_audio(pcm_path)?;
    if audio.len() < MAX_PENDING_SAMPLES + INPUT_CHUNK_SAMPLES {
        return Err(format!(
            "contract fixture must contain at least {} samples",
            MAX_PENDING_SAMPLES + INPUT_CHUNK_SAMPLES
        ));
    }

    let session = create_session(&model_directory, 480, 16, device_kind)?;
    let initial = step(session.handle);
    if initial.status != NEEDS_AUDIO || initial.json.is_none() {
        return Err(format!(
            "empty open session returned status {}, expected NEEDS_AUDIO ({NEEDS_AUDIO})",
            initial.status
        ));
    }
    for chunk in audio[..MAX_PENDING_SAMPLES].chunks(INPUT_CHUNK_SAMPLES) {
        let response = feed(session.handle, chunk);
        if response.status != OK {
            return Err(response
                .error
                .unwrap_or_else(|| format!("queue fill returned status {}", response.status)));
        }
    }
    let backpressure = feed(
        session.handle,
        &audio[MAX_PENDING_SAMPLES..MAX_PENDING_SAMPLES + INPUT_CHUNK_SAMPLES],
    );
    if backpressure.status != BACKPRESSURE {
        return Err(format!(
            "full queue returned status {}, expected BACKPRESSURE ({BACKPRESSURE})",
            backpressure.status
        ));
    }

    let closed = close(session.handle);
    if closed.status != OK {
        return Err(closed
            .error
            .unwrap_or_else(|| format!("close returned status {}", closed.status)));
    }

    let closed_feed = feed(session.handle, &audio[..1]);
    if closed_feed.status != INVALID_ARGUMENT {
        return Err(format!(
            "feed after close returned status {}, expected INVALID_ARGUMENT ({INVALID_ARGUMENT})",
            closed_feed.status
        ));
    }
    unsafe { cuttledoc_voxtral_mlx_session_cancel(session.handle) };
    let cancelled = step(session.handle);
    if cancelled.status != CANCELLED || cancelled.json.is_some() {
        return Err(format!(
            "cancelled step returned status {}, expected CANCELLED ({CANCELLED})",
            cancelled.status
        ));
    }

    println!(
        "{{\"status\":\"ok\",\"boundary\":\"repository-owned-rust-c-abi-over-official-mlx\",\"stage\":\"voxtral-streaming-lifecycle\",\"device\":{},\"input_chunk_samples\":{},\"queue_capacity_samples\":{},\"max_ingest_samples_per_step\":{},\"stable_statuses\":{{\"invalid_argument\":{},\"cancelled\":{},\"backpressure\":{},\"needs_audio\":{},\"done\":{}}},\"capabilities\":{{\"bounded_ingestion\":true,\"backpressure\":true,\"cancellation\":true,\"persistent_model_state\":true,\"transcription\":true}}}}",
        json_string(device),
        INPUT_CHUNK_SAMPLES,
        MAX_PENDING_SAMPLES,
        MAX_INGEST_SAMPLES_PER_STEP,
        INVALID_ARGUMENT,
        CANCELLED,
        BACKPRESSURE,
        NEEDS_AUDIO,
        DONE,
    );
    Ok(())
}

struct ProducerStats {
    audio_close_ms: f64,
    backpressure_count: usize,
    chunk_count: usize,
    maximum_schedule_lateness_ms: f64,
}

struct AppendEvent {
    elapsed_ms: f64,
    audio_fed_ms: f64,
    delta: String,
}

struct ProducerThread {
    handle: *mut c_void,
    thread: Option<thread::JoinHandle<Result<ProducerStats, String>>>,
}

impl ProducerThread {
    fn join(mut self) -> Result<ProducerStats, String> {
        self.thread
            .take()
            .expect("producer thread is present")
            .join()
            .map_err(|_| "audio producer thread panicked".to_owned())?
    }
}

impl Drop for ProducerThread {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            unsafe { cuttledoc_voxtral_mlx_session_cancel(self.handle) };
            let _ = thread.join();
        }
    }
}

fn stream(
    model_directory: &str,
    pcm_path: &str,
    delay_ms: &str,
    chunk_ms: &str,
    max_decode_tokens: &str,
    device: &str,
) -> Result<(), String> {
    let model_directory = c_string(model_directory, "model path")?;
    let audio = Arc::new(read_audio(pcm_path)?);
    let delay_ms = parse_positive_i32(delay_ms, "delay")?;
    let chunk_ms = parse_positive_usize(chunk_ms, "chunk milliseconds")?;
    let max_decode_tokens = parse_positive_usize(max_decode_tokens, "max decode tokens")?;
    let chunk_samples = chunk_ms
        .checked_mul(SAMPLE_RATE_HZ)
        .ok_or_else(|| "chunk size overflows usize".to_owned())?
        / 1_000;
    if chunk_samples == 0 || chunk_samples > MAX_PENDING_SAMPLES {
        return Err(format!(
            "chunk must resolve to 1..={MAX_PENDING_SAMPLES} samples"
        ));
    }
    let device_kind = parse_device(device)?;
    let session = create_session(&model_directory, delay_ms, max_decode_tokens, device_kind)?;
    let initial = step(session.handle);
    if initial.status != NEEDS_AUDIO || initial.json.is_none() {
        return Err(format!(
            "empty open session returned status {}, expected NEEDS_AUDIO ({NEEDS_AUDIO})",
            initial.status
        ));
    }

    let started = Instant::now();
    let fed_samples = Arc::new(AtomicUsize::new(0));
    let producer_finished = Arc::new(AtomicBool::new(false));
    let producer_fed_samples = Arc::clone(&fed_samples);
    let producer_finished_signal = Arc::clone(&producer_finished);
    let producer_audio = Arc::clone(&audio);
    let producer_handle = session.handle as usize;
    let producer = ProducerThread {
        handle: session.handle,
        thread: Some(thread::spawn(move || {
            let result = produce_audio(
                producer_handle,
                producer_audio,
                chunk_samples,
                producer_fed_samples,
                started,
            );
            producer_finished_signal.store(true, Ordering::Release);
            result
        })),
    };

    let mut step_call_count = 0usize;
    let mut total_ingested = 0usize;
    let mut maximum_ingested = 0usize;
    let mut maximum_step_wall_ms = 0.0_f64;
    let mut maximum_mlx_elapsed_ms = 0.0_f64;
    let mut previous_text = String::new();
    let mut append_events = Vec::new();
    let mut wait_after_fed_samples = None;

    let (final_text, generated_tokens, adapter_frames) = loop {
        if let Some(observed) = wait_after_fed_samples {
            while fed_samples.load(Ordering::Acquire) == observed
                && !producer_finished.load(Ordering::Acquire)
            {
                thread::sleep(Duration::from_millis(1));
            }
            wait_after_fed_samples = None;
        }
        let step_started = Instant::now();
        let response = step(session.handle);
        let step_wall_ms = step_started.elapsed().as_secs_f64() * 1_000.0;
        maximum_step_wall_ms = maximum_step_wall_ms.max(step_wall_ms);
        step_call_count += 1;
        match response.status {
            OK | NEEDS_AUDIO | DONE => {
                let json = response
                    .json
                    .as_deref()
                    .ok_or_else(|| format!("status {} returned no JSON", response.status))?;
                let ingested = json_usize(json, "ingested_samples")?;
                if ingested > MAX_INGEST_SAMPLES_PER_STEP {
                    return Err(format!(
                        "step ingested {ingested} samples, exceeding {MAX_INGEST_SAMPLES_PER_STEP}"
                    ));
                }
                total_ingested += ingested;
                maximum_ingested = maximum_ingested.max(ingested);
                maximum_mlx_elapsed_ms =
                    maximum_mlx_elapsed_ms.max(json_number(json, "mlx_elapsed_ms")?);
                let current_generated_tokens = json_usize(json, "generated_tokens")?;
                let current_adapter_frames = json_usize(json, "adapter_frames")?;
                let text = json_string_field(json, "text")?;
                let delta = json_string_field(json, "text_delta")?;
                if !text.starts_with(&previous_text) {
                    return Err("streaming text revoked an already emitted prefix".to_owned());
                }
                let expected_delta = &text[previous_text.len()..];
                if delta != expected_delta {
                    return Err(format!(
                        "text_delta mismatch: expected {}, got {}",
                        json_string(expected_delta),
                        json_string(&delta)
                    ));
                }
                if !delta.is_empty() {
                    append_events.push(AppendEvent {
                        elapsed_ms: started.elapsed().as_secs_f64() * 1_000.0,
                        audio_fed_ms: fed_samples.load(Ordering::Acquire) as f64
                            / SAMPLE_RATE_HZ as f64
                            * 1_000.0,
                        delta,
                    });
                }
                previous_text = text.clone();
                if response.status == DONE {
                    break (text, current_generated_tokens, current_adapter_frames);
                }
                if response.status == NEEDS_AUDIO {
                    wait_after_fed_samples = Some(fed_samples.load(Ordering::Acquire));
                }
            }
            status => {
                return Err(response
                    .error
                    .unwrap_or_else(|| format!("streaming step returned status {status}")));
            }
        }
    };
    let final_ms = started.elapsed().as_secs_f64() * 1_000.0;
    let producer = producer.join()?;
    if total_ingested != audio.len() {
        return Err(format!(
            "stream ingested {total_ingested} of {} fed samples",
            audio.len()
        ));
    }
    if final_text.is_empty() || append_events.is_empty() {
        return Err("streaming session completed without transcript updates".to_owned());
    }
    let first_append = &append_events[0];
    let endpoint_finalization_ms = final_ms - producer.audio_close_ms;

    print!(
        "{{\"schema_version\":\"1.0.0\",\"status\":\"ok\",\"boundary\":\"repository-owned-rust-c-abi-over-official-mlx\",\"stage\":\"voxtral-incremental-streaming\",\"device\":{},\"transcription_delay_ms\":{},\"fixture\":{{\"pcm_samples\":{},\"duration_ms\":{:.6}}},\"procedure\":{{\"realtime_pacing\":true,\"producer_thread_independent_from_mlx_executor\":true,\"chunk_ms\":{},\"chunk_samples\":{},\"max_pending_samples\":{},\"max_ingest_samples_per_step\":{},\"max_decode_tokens_per_step\":{}}},\"timing\":{{\"first_append_ms\":{:.6},\"first_append_audio_fed_ms\":{:.6},\"audio_close_ms\":{:.6},\"final_ms\":{:.6},\"endpoint_finalization_ms\":{:.6},\"maximum_step_wall_ms\":{:.6},\"maximum_mlx_elapsed_ms\":{:.6},\"maximum_feed_schedule_lateness_ms\":{:.6}}},\"streaming\":{{\"append_only\":true,\"update_count\":{},\"revoke_count\":0,\"audio_chunk_count\":{},\"step_call_count\":{},\"generated_tokens\":{},\"adapter_frames\":{},\"total_ingested_samples\":{},\"maximum_ingested_samples\":{},\"backpressure_count\":{},\"done\":true}},\"text\":{},\"events\":[",
        json_string(device),
        delay_ms,
        audio.len(),
        audio.len() as f64 / SAMPLE_RATE_HZ as f64 * 1_000.0,
        chunk_ms,
        chunk_samples,
        MAX_PENDING_SAMPLES,
        MAX_INGEST_SAMPLES_PER_STEP,
        max_decode_tokens,
        first_append.elapsed_ms,
        first_append.audio_fed_ms,
        producer.audio_close_ms,
        final_ms,
        endpoint_finalization_ms,
        maximum_step_wall_ms,
        maximum_mlx_elapsed_ms,
        producer.maximum_schedule_lateness_ms,
        append_events.len(),
        producer.chunk_count,
        step_call_count,
        generated_tokens,
        adapter_frames,
        total_ingested,
        maximum_ingested,
        producer.backpressure_count,
        json_string(&final_text),
    );
    for (index, event) in append_events.iter().enumerate() {
        if index != 0 {
            print!(",");
        }
        print!(
            "{{\"index\":{},\"elapsed_ms\":{:.6},\"audio_fed_ms\":{:.6},\"delta\":{}}}",
            index,
            event.elapsed_ms,
            event.audio_fed_ms,
            json_string(&event.delta)
        );
    }
    println!("]}}");
    Ok(())
}

fn produce_audio(
    handle: usize,
    audio: Arc<Vec<f32>>,
    chunk_samples: usize,
    fed_samples: Arc<AtomicUsize>,
    started: Instant,
) -> Result<ProducerStats, String> {
    let handle = handle as *mut c_void;
    let mut offset = 0usize;
    let mut chunk_count = 0usize;
    let mut backpressure_count = 0usize;
    let mut maximum_schedule_lateness_ms = 0.0_f64;
    while offset < audio.len() {
        let end = (offset + chunk_samples).min(audio.len());
        let target = Duration::from_secs_f64(end as f64 / SAMPLE_RATE_HZ as f64);
        if let Some(remaining) = target.checked_sub(started.elapsed()) {
            thread::sleep(remaining);
        }
        maximum_schedule_lateness_ms = maximum_schedule_lateness_ms.max(
            (started.elapsed().as_secs_f64() - target.as_secs_f64()).max(0.0) * 1_000.0,
        );
        loop {
            let response = feed(handle, &audio[offset..end]);
            match response.status {
                OK => break,
                BACKPRESSURE => {
                    backpressure_count += 1;
                    thread::sleep(Duration::from_millis(1));
                }
                status => {
                    return Err(response
                        .error
                        .unwrap_or_else(|| format!("audio producer feed returned status {status}")));
                }
            }
        }
        offset = end;
        chunk_count += 1;
        fed_samples.store(offset, Ordering::Release);
    }
    let response = close(handle);
    if response.status != OK {
        return Err(response
            .error
            .unwrap_or_else(|| format!("audio producer close returned status {}", response.status)));
    }
    Ok(ProducerStats {
        audio_close_ms: started.elapsed().as_secs_f64() * 1_000.0,
        backpressure_count,
        chunk_count,
        maximum_schedule_lateness_ms,
    })
}

fn create_session(
    model_directory: &CString,
    delay_ms: i32,
    max_decode_tokens_per_step: usize,
    device_kind: i32,
) -> Result<Session, String> {
    let mut status = OK;
    let mut error = ptr::null_mut();
    let handle = unsafe {
        cuttledoc_voxtral_mlx_session_create(
            model_directory.as_ptr(),
            delay_ms,
            MAX_GENERATED_TOKENS,
            max_decode_tokens_per_step,
            device_kind,
            MAX_PENDING_SAMPLES,
            MAX_INGEST_SAMPLES_PER_STEP,
            &mut status,
            &mut error,
        )
    };
    let error = take_string(error);
    if handle.is_null() || status != OK {
        return Err(error.unwrap_or_else(|| {
            format!("MLX session creation returned status {status} without a message")
        }));
    }
    Ok(Session { handle })
}

fn feed(handle: *mut c_void, audio: &[f32]) -> Response {
    let mut error = ptr::null_mut();
    let status = unsafe {
        cuttledoc_voxtral_mlx_session_feed(handle, audio.as_ptr(), audio.len(), &mut error)
    };
    Response {
        status,
        json: None,
        error: take_string(error),
    }
}

fn close(handle: *mut c_void) -> Response {
    let mut error = ptr::null_mut();
    let status = unsafe { cuttledoc_voxtral_mlx_session_close(handle, &mut error) };
    Response {
        status,
        json: None,
        error: take_string(error),
    }
}

fn step(handle: *mut c_void) -> Response {
    let mut json = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe { cuttledoc_voxtral_mlx_session_step(handle, &mut json, &mut error) };
    Response {
        status,
        json: take_string(json),
        error: take_string(error),
    }
}

fn read_audio(pcm_path: &str) -> Result<Vec<f32>, String> {
    let bytes = fs::read(pcm_path)
        .map_err(|error| format!("could not read PCM fixture {pcm_path}: {error}"))?;
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err("PCM fixture must contain non-empty little-endian float32".to_owned());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte PCM chunk")))
        .collect())
}

fn parse_device(device: &str) -> Result<i32, String> {
    match device {
        "cpu" => Ok(0),
        "gpu" => Ok(1),
        _ => Err("device must be cpu or gpu".to_owned()),
    }
}

fn parse_positive_i32(value: &str, name: &str) -> Result<i32, String> {
    let value = value
        .parse::<i32>()
        .map_err(|error| format!("{name} must be a positive integer: {error}"))?;
    if value <= 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(value)
}

fn parse_positive_usize(value: &str, name: &str) -> Result<usize, String> {
    let value = value
        .parse::<usize>()
        .map_err(|error| format!("{name} must be a positive integer: {error}"))?;
    if value == 0 {
        return Err(format!("{name} must be positive"));
    }
    Ok(value)
}

fn c_string(value: &str, name: &str) -> Result<CString, String> {
    CString::new(value).map_err(|_| format!("{name} contains an embedded NUL byte"))
}

fn json_number(json: &str, key: &str) -> Result<f64, String> {
    let marker = format!("\"{key}\":");
    let start = json
        .find(&marker)
        .ok_or_else(|| format!("step JSON is missing {key}"))?
        + marker.len();
    let end = json[start..]
        .find(|character: char| {
            !character.is_ascii_digit()
                && character != '-'
                && character != '+'
                && character != '.'
                && character != 'e'
                && character != 'E'
        })
        .map(|end| start + end)
        .unwrap_or(json.len());
    json[start..end]
        .parse::<f64>()
        .map_err(|error| format!("could not parse {key} from step JSON: {error}"))
}

fn json_usize(json: &str, key: &str) -> Result<usize, String> {
    let value = json_number(json, key)?;
    if value < 0.0 || value.fract() != 0.0 || value > usize::MAX as f64 {
        return Err(format!("step JSON {key} is not a usize: {value}"));
    }
    Ok(value as usize)
}

fn json_string_field(json: &str, key: &str) -> Result<String, String> {
    let marker = format!("\"{key}\":\"");
    let start = json
        .find(&marker)
        .ok_or_else(|| format!("step JSON is missing string field {key}"))?
        + marker.len();
    let mut characters = json[start..].chars();
    let mut result = String::new();
    while let Some(character) = characters.next() {
        match character {
            '"' => return Ok(result),
            '\\' => match characters
                .next()
                .ok_or_else(|| format!("unterminated escape in JSON field {key}"))?
            {
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                '/' => result.push('/'),
                'b' => result.push('\u{0008}'),
                'f' => result.push('\u{000c}'),
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                'u' => {
                    let digits = (0..4)
                        .map(|_| {
                            characters.next().ok_or_else(|| {
                                format!("short Unicode escape in JSON field {key}")
                            })
                        })
                        .collect::<Result<String, String>>()?;
                    let value = u32::from_str_radix(&digits, 16).map_err(|error| {
                        format!("invalid Unicode escape in JSON field {key}: {error}")
                    })?;
                    result.push(char::from_u32(value).ok_or_else(|| {
                        format!("invalid Unicode scalar in JSON field {key}")
                    })?);
                }
                escape => {
                    return Err(format!(
                        "unsupported JSON escape \\{escape} in field {key}"
                    ));
                }
            },
            character => result.push(character),
        }
    }
    Err(format!("unterminated JSON string field {key}"))
}

fn json_string(value: &str) -> String {
    let mut result = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            character if character <= '\u{1f}' => {
                result.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => result.push(character),
        }
    }
    result.push('"');
    result
}

fn take_string(value: *mut c_char) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let result = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe { cuttledoc_voxtral_mlx_free_string(value) };
    Some(result)
}

fn usage() -> String {
    "usage:\n  cuttledoc-voxtral-mlx inspect MODEL_DIR\n  cuttledoc-voxtral-mlx frontend MODEL_DIR PCM_F32LE DELAY_MS cpu|gpu\n  cuttledoc-voxtral-mlx encoder MODEL_DIR PCM_F32LE DELAY_MS cpu|gpu\n  cuttledoc-voxtral-mlx transcribe MODEL_DIR PCM_F32LE DELAY_MS MAX_TOKENS cpu|gpu\n  cuttledoc-voxtral-mlx contract MODEL_DIR PCM_F32LE cpu|gpu\n  cuttledoc-voxtral-mlx stream MODEL_DIR PCM_F32LE DELAY_MS CHUNK_MS MAX_DECODE_TOKENS cpu|gpu".to_owned()
}
