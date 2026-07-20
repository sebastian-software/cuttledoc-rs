use std::{
    env,
    ffi::{CStr, CString, c_char, c_void},
    fs, ptr,
    time::Instant,
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
    fn cuttledoc_voxtral_mlx_session_create(
        model_directory: *const c_char,
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
    fn cuttledoc_voxtral_mlx_session_close(
        handle: *mut c_void,
        error_out: *mut *mut c_char,
    ) -> i32;
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
        [command, model_directory, pcm_path, device] if command == "contract" => {
            contract(model_directory, pcm_path, device)
        }
        _ => Err(usage()),
    }
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
    if audio.len() <= MAX_PENDING_SAMPLES {
        return Err(format!(
            "contract fixture must exceed the {}-sample queue capacity",
            MAX_PENDING_SAMPLES
        ));
    }

    let session = create_session(&model_directory, device_kind)?;
    let initial = step(session.handle);
    if initial.status != NEEDS_AUDIO || initial.json.is_none() {
        return Err(format!(
            "empty open session returned status {}, expected NEEDS_AUDIO ({NEEDS_AUDIO})",
            initial.status
        ));
    }

    let expected_energy = audio
        .iter()
        .map(|sample| {
            let sample = f64::from(*sample);
            sample * sample
        })
        .sum::<f64>();
    let mut offset = 0;
    let mut backpressure_count = 0;
    let mut total_ingested = 0;
    let mut actual_energy = 0.0;
    let mut step_count = 0;
    let mut maximum_ingested = 0;
    let mut maximum_step_wall_ms = 0.0_f64;
    let mut maximum_mlx_elapsed_ms = 0.0_f64;

    while offset < audio.len() {
        let end = (offset + INPUT_CHUNK_SAMPLES).min(audio.len());
        let response = feed(session.handle, &audio[offset..end]);
        match response.status {
            OK => offset = end,
            BACKPRESSURE => {
                backpressure_count += 1;
                consume_step(
                    session.handle,
                    &mut total_ingested,
                    &mut actual_energy,
                    &mut step_count,
                    &mut maximum_ingested,
                    &mut maximum_step_wall_ms,
                    &mut maximum_mlx_elapsed_ms,
                )?;
            }
            status => {
                return Err(response.error.unwrap_or_else(|| {
                    format!("feed returned unexpected status {status}")
                }));
            }
        }
    }
    if backpressure_count == 0 {
        return Err("fixture did not exercise hard queue backpressure".to_owned());
    }

    let closed = close(session.handle);
    if closed.status != OK {
        return Err(closed
            .error
            .unwrap_or_else(|| format!("close returned status {}", closed.status)));
    }
    loop {
        let started = Instant::now();
        let response = step(session.handle);
        let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
        maximum_step_wall_ms = maximum_step_wall_ms.max(wall_ms);
        match response.status {
            OK => record_step(
                response.json.as_deref(),
                &mut total_ingested,
                &mut actual_energy,
                &mut step_count,
                &mut maximum_ingested,
                &mut maximum_mlx_elapsed_ms,
            )?,
            DONE if response.json.is_some() => break,
            status => {
                return Err(response.error.unwrap_or_else(|| {
                    format!("drain step returned unexpected status {status}")
                }));
            }
        }
    }

    if total_ingested != audio.len() || maximum_ingested > MAX_INGEST_SAMPLES_PER_STEP {
        return Err(format!(
            "bounded ingestion mismatch: fed {}, ingested {}, maximum step {}",
            audio.len(), total_ingested, maximum_ingested
        ));
    }
    let energy_absolute_error = (actual_energy - expected_energy).abs();
    let energy_relative_error = energy_absolute_error / expected_energy.max(f64::EPSILON);
    if energy_relative_error > 5e-5 {
        return Err(format!(
            "official-MLX energy fingerprint drifted: expected {expected_energy}, got {actual_energy}, relative error {energy_relative_error}"
        ));
    }

    let closed_feed = feed(session.handle, &audio[..1]);
    if closed_feed.status != INVALID_ARGUMENT {
        return Err(format!(
            "feed after close returned status {}, expected INVALID_ARGUMENT ({INVALID_ARGUMENT})",
            closed_feed.status
        ));
    }
    drop(session);

    let cancelled_session = create_session(&model_directory, device_kind)?;
    unsafe { cuttledoc_voxtral_mlx_session_cancel(cancelled_session.handle) };
    let cancelled = step(cancelled_session.handle);
    if cancelled.status != CANCELLED || cancelled.json.is_some() {
        return Err(format!(
            "cancelled step returned status {}, expected CANCELLED ({CANCELLED})",
            cancelled.status
        ));
    }

    println!(
        "{{\"status\":\"ok\",\"boundary\":\"repository-owned-rust-c-abi-over-official-mlx\",\"stage\":\"voxtral-bounded-ingestion\",\"device\":{},\"pcm_samples\":{},\"input_chunk_samples\":{},\"queue_capacity_samples\":{},\"max_ingest_samples_per_step\":{},\"total_fed_samples\":{},\"total_ingested_samples\":{},\"step_count\":{},\"maximum_ingested_samples\":{},\"backpressure_count\":{},\"stable_statuses\":{{\"invalid_argument\":{},\"cancelled\":{},\"backpressure\":{},\"needs_audio\":{},\"done\":{}}},\"maximum_step_wall_ms\":{:.6},\"maximum_mlx_elapsed_ms\":{:.6},\"energy\":{{\"cpu_expected_sum_squares\":{:.17},\"mlx_sum_squares\":{:.17},\"relative_error\":{:.17}}},\"capabilities\":{{\"bounded_ingestion\":true,\"backpressure\":true,\"cancellation\":true,\"transcription\":false}}}}",
        json_string(device),
        audio.len(),
        INPUT_CHUNK_SAMPLES,
        MAX_PENDING_SAMPLES,
        MAX_INGEST_SAMPLES_PER_STEP,
        audio.len(),
        total_ingested,
        step_count,
        maximum_ingested,
        backpressure_count,
        INVALID_ARGUMENT,
        CANCELLED,
        BACKPRESSURE,
        NEEDS_AUDIO,
        DONE,
        maximum_step_wall_ms,
        maximum_mlx_elapsed_ms,
        expected_energy,
        actual_energy,
        energy_relative_error,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn consume_step(
    handle: *mut c_void,
    total_ingested: &mut usize,
    actual_energy: &mut f64,
    step_count: &mut usize,
    maximum_ingested: &mut usize,
    maximum_step_wall_ms: &mut f64,
    maximum_mlx_elapsed_ms: &mut f64,
) -> Result<(), String> {
    let started = Instant::now();
    let response = step(handle);
    *maximum_step_wall_ms = maximum_step_wall_ms
        .max(started.elapsed().as_secs_f64() * 1_000.0);
    if response.status != OK {
        return Err(response.error.unwrap_or_else(|| {
            format!(
                "backpressure drain returned status {}, expected OK",
                response.status
            )
        }));
    }
    record_step(
        response.json.as_deref(),
        total_ingested,
        actual_energy,
        step_count,
        maximum_ingested,
        maximum_mlx_elapsed_ms,
    )
}

fn record_step(
    json: Option<&str>,
    total_ingested: &mut usize,
    actual_energy: &mut f64,
    step_count: &mut usize,
    maximum_ingested: &mut usize,
    maximum_mlx_elapsed_ms: &mut f64,
) -> Result<(), String> {
    let json = json.ok_or_else(|| "successful step returned no JSON".to_owned())?;
    let ingested = json_usize(json, "ingested_samples")?;
    let energy = json_number(json, "mlx_sum_squares")?;
    let mlx_elapsed_ms = json_number(json, "mlx_elapsed_ms")?;
    if ingested == 0 || ingested > MAX_INGEST_SAMPLES_PER_STEP {
        return Err(format!("step ingested invalid bounded slice of {ingested} samples"));
    }
    *total_ingested += ingested;
    *actual_energy += energy;
    *step_count += 1;
    *maximum_ingested = (*maximum_ingested).max(ingested);
    *maximum_mlx_elapsed_ms = (*maximum_mlx_elapsed_ms).max(mlx_elapsed_ms);
    Ok(())
}

fn create_session(model_directory: &CString, device_kind: i32) -> Result<Session, String> {
    let mut status = OK;
    let mut error = ptr::null_mut();
    let handle = unsafe {
        cuttledoc_voxtral_mlx_session_create(
            model_directory.as_ptr(),
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
    let status = unsafe {
        cuttledoc_voxtral_mlx_session_step(handle, &mut json, &mut error)
    };
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
    "usage:\n  cuttledoc-voxtral-mlx inspect MODEL_DIR\n  cuttledoc-voxtral-mlx frontend MODEL_DIR PCM_F32LE DELAY_MS cpu|gpu\n  cuttledoc-voxtral-mlx contract MODEL_DIR PCM_F32LE cpu|gpu".to_owned()
}
