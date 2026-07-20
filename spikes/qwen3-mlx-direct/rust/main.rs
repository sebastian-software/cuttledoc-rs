use std::{
    env,
    ffi::{CStr, CString, c_char, c_void},
    fs, ptr,
    sync::{Arc, Barrier},
    thread,
    time::{Duration, Instant},
};

unsafe extern "C" {
    fn cuttledoc_qwen3_mlx_inspect_model(
        model_directory: *const c_char,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_qwen3_mlx_probe_audio_frontend(
        model_directory: *const c_char,
        audio: *const f32,
        audio_len: usize,
        device_kind: i32,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_qwen3_mlx_probe_audio_encoder(
        model_directory: *const c_char,
        audio: *const f32,
        audio_len: usize,
        device_kind: i32,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_qwen3_mlx_probe_prompt_embeddings(
        model_directory: *const c_char,
        audio: *const f32,
        audio_len: usize,
        language: *const c_char,
        device_kind: i32,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_qwen3_mlx_probe_decoder_prefill(
        model_directory: *const c_char,
        audio: *const f32,
        audio_len: usize,
        language: *const c_char,
        device_kind: i32,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_qwen3_mlx_transcribe(
        model_directory: *const c_char,
        audio: *const f32,
        audio_len: usize,
        language: *const c_char,
        device_kind: i32,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_qwen3_mlx_session_create(
        model_directory: *const c_char,
        device_kind: i32,
        status_out: *mut i32,
        error_out: *mut *mut c_char,
    ) -> *mut c_void;
    fn cuttledoc_qwen3_mlx_session_transcribe(
        handle: *mut c_void,
        audio: *const f32,
        audio_len: usize,
        language: *const c_char,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_qwen3_mlx_session_cancel(handle: *mut c_void);
    fn cuttledoc_qwen3_mlx_session_destroy(handle: *mut c_void);
    fn cuttledoc_qwen3_mlx_free_string(value: *mut c_char);
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
        [model_directory] => inspect(model_directory),
        [command, model_directory, pcm_path, device] if command == "frontend" => {
            probe_audio(model_directory, pcm_path, device, AudioProbe::Frontend)
        }
        [command, model_directory, pcm_path, device] if command == "encoder" => {
            probe_audio(model_directory, pcm_path, device, AudioProbe::Encoder)
        }
        [command, model_directory, pcm_path, language, device] if command == "prompt" => {
            probe_text_boundary(
                model_directory,
                pcm_path,
                language,
                device,
                TextProbe::Prompt,
            )
        }
        [command, model_directory, pcm_path, language, device] if command == "decoder" => {
            probe_text_boundary(
                model_directory,
                pcm_path,
                language,
                device,
                TextProbe::Decoder,
            )
        }
        [command, model_directory, pcm_path, language, device] if command == "transcribe" => {
            probe_text_boundary(
                model_directory,
                pcm_path,
                language,
                device,
                TextProbe::Transcribe,
            )
        }
        [
            command,
            model_directory,
            pcm_path,
            language,
            device,
            lifecycle_count,
            runs_per_lifecycle,
        ] if command == "lifecycle" => lifecycle(
            model_directory,
            pcm_path,
            language,
            device,
            lifecycle_count,
            runs_per_lifecycle,
        ),
        [command, model_directory, pcm_path, language, device] if command == "cancel" => {
            cancel(model_directory, pcm_path, language, device)
        }
        _ => Err(usage()),
    }
}

fn inspect(model_directory: &str) -> Result<(), String> {
    let model_directory = CString::new(model_directory)
        .map_err(|_| "model path contains an embedded NUL byte".to_owned())?;

    let mut json = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        cuttledoc_qwen3_mlx_inspect_model(model_directory.as_ptr(), &mut json, &mut error)
    };
    if status != 0 {
        if !json.is_null() {
            unsafe { cuttledoc_qwen3_mlx_free_string(json) };
        }
        return Err(take_string(error)
            .unwrap_or_else(|| "MLX shim returned an error without a message".to_owned()));
    }
    let result =
        take_string(json).ok_or_else(|| "MLX shim returned success without JSON".to_owned())?;
    println!("{result}");
    Ok(())
}

#[derive(Clone, Copy)]
enum AudioProbe {
    Frontend,
    Encoder,
}

fn probe_audio(
    model_directory: &str,
    pcm_path: &str,
    device: &str,
    probe: AudioProbe,
) -> Result<(), String> {
    let model_directory = CString::new(model_directory)
        .map_err(|_| "model path contains an embedded NUL byte".to_owned())?;
    let device_kind = match device {
        "cpu" => 0,
        "gpu" => 1,
        _ => return Err("device must be cpu or gpu".to_owned()),
    };
    let pcm_bytes =
        fs::read(pcm_path).map_err(|error| format!("could not read PCM fixture: {error}"))?;
    if pcm_bytes.is_empty() || pcm_bytes.len() % 4 != 0 {
        return Err("PCM fixture must contain non-empty little-endian float32".to_owned());
    }
    let audio = pcm_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte PCM chunk")))
        .collect::<Vec<_>>();

    let mut json = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        match probe {
            AudioProbe::Frontend => cuttledoc_qwen3_mlx_probe_audio_frontend(
                model_directory.as_ptr(),
                audio.as_ptr(),
                audio.len(),
                device_kind,
                &mut json,
                &mut error,
            ),
            AudioProbe::Encoder => cuttledoc_qwen3_mlx_probe_audio_encoder(
                model_directory.as_ptr(),
                audio.as_ptr(),
                audio.len(),
                device_kind,
                &mut json,
                &mut error,
            ),
        }
    };
    if status != 0 {
        if !json.is_null() {
            unsafe { cuttledoc_qwen3_mlx_free_string(json) };
        }
        return Err(take_string(error)
            .unwrap_or_else(|| "MLX frontend returned an error without a message".to_owned()));
    }
    let result =
        take_string(json).ok_or_else(|| "MLX frontend returned success without JSON".to_owned())?;
    println!("{result}");
    Ok(())
}

#[derive(Clone, Copy)]
enum TextProbe {
    Prompt,
    Decoder,
    Transcribe,
}

fn probe_text_boundary(
    model_directory: &str,
    pcm_path: &str,
    language: &str,
    device: &str,
    probe: TextProbe,
) -> Result<(), String> {
    let model_directory = CString::new(model_directory)
        .map_err(|_| "model path contains an embedded NUL byte".to_owned())?;
    let language =
        CString::new(language).map_err(|_| "language contains an embedded NUL byte".to_owned())?;
    let device_kind = match device {
        "cpu" => 0,
        "gpu" => 1,
        _ => return Err("device must be cpu or gpu".to_owned()),
    };
    let pcm_bytes =
        fs::read(pcm_path).map_err(|error| format!("could not read PCM fixture: {error}"))?;
    if pcm_bytes.is_empty() || pcm_bytes.len() % 4 != 0 {
        return Err("PCM fixture must contain non-empty little-endian float32".to_owned());
    }
    let audio = pcm_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte PCM chunk")))
        .collect::<Vec<_>>();

    let mut json = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        match probe {
            TextProbe::Prompt => cuttledoc_qwen3_mlx_probe_prompt_embeddings(
                model_directory.as_ptr(),
                audio.as_ptr(),
                audio.len(),
                language.as_ptr(),
                device_kind,
                &mut json,
                &mut error,
            ),
            TextProbe::Decoder => cuttledoc_qwen3_mlx_probe_decoder_prefill(
                model_directory.as_ptr(),
                audio.as_ptr(),
                audio.len(),
                language.as_ptr(),
                device_kind,
                &mut json,
                &mut error,
            ),
            TextProbe::Transcribe => cuttledoc_qwen3_mlx_transcribe(
                model_directory.as_ptr(),
                audio.as_ptr(),
                audio.len(),
                language.as_ptr(),
                device_kind,
                &mut json,
                &mut error,
            ),
        }
    };
    if status != 0 {
        if !json.is_null() {
            unsafe { cuttledoc_qwen3_mlx_free_string(json) };
        }
        return Err(take_string(error).unwrap_or_else(|| {
            "MLX text-boundary probe returned an error without a message".to_owned()
        }));
    }
    let result = take_string(json)
        .ok_or_else(|| "MLX text-boundary probe returned success without JSON".to_owned())?;
    println!("{result}");
    Ok(())
}

struct MlxSession {
    handle: *mut c_void,
}

impl Drop for MlxSession {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { cuttledoc_qwen3_mlx_session_destroy(self.handle) };
        }
    }
}

struct SessionResponse {
    status: i32,
    json: Option<String>,
    error: Option<String>,
}

fn lifecycle(
    model_directory: &str,
    pcm_path: &str,
    language: &str,
    device: &str,
    lifecycle_count: &str,
    runs_per_lifecycle: &str,
) -> Result<(), String> {
    let model_directory = CString::new(model_directory)
        .map_err(|_| "model path contains an embedded NUL byte".to_owned())?;
    let language =
        CString::new(language).map_err(|_| "language contains an embedded NUL byte".to_owned())?;
    let device_kind = parse_device(device)?;
    let audio = read_audio(pcm_path)?;
    let lifecycle_count = parse_positive(lifecycle_count, "lifecycle count")?;
    let runs_per_lifecycle = parse_positive(runs_per_lifecycle, "runs per lifecycle")?;

    let mut invalid_argument = None;
    let mut sessions = Vec::with_capacity(lifecycle_count);
    for lifecycle_index in 0..lifecycle_count {
        let create_started = Instant::now();
        let mut session = create_session(&model_directory, device_kind)?;
        let create_ms = create_started.elapsed().as_secs_f64() * 1_000.0;

        if lifecycle_index == 0 {
            let response = session_call(session.handle, ptr::null(), 0, language.as_ptr());
            if response.status != 1 || response.json.is_some() {
                return Err(format!(
                    "invalid-argument probe returned status {}, expected 1",
                    response.status
                ));
            }
            invalid_argument = Some(response);
        }

        let mut runs = Vec::with_capacity(runs_per_lifecycle);
        for run_index in 0..runs_per_lifecycle {
            let started = Instant::now();
            let response = session_call(
                session.handle,
                audio.as_ptr(),
                audio.len(),
                language.as_ptr(),
            );
            let wall_ms = started.elapsed().as_secs_f64() * 1_000.0;
            if response.status != 0 {
                return Err(response.error.unwrap_or_else(|| {
                    format!("session transcription returned status {}", response.status)
                }));
            }
            let transcription = response
                .json
                .ok_or_else(|| "session transcription returned success without JSON".to_owned())?;
            runs.push(format!(
                "{{\"run_index\":{run_index},\"wall_ms\":{wall_ms:.6},\"transcription\":{transcription}}}"
            ));
        }

        let destroy_started = Instant::now();
        unsafe { cuttledoc_qwen3_mlx_session_destroy(session.handle) };
        session.handle = ptr::null_mut();
        let destroy_ms = destroy_started.elapsed().as_secs_f64() * 1_000.0;
        sessions.push(format!(
            "{{\"lifecycle_index\":{lifecycle_index},\"create_ms\":{create_ms:.6},\"destroy_ms\":{destroy_ms:.6},\"runs\":[{}]}}",
            runs.join(",")
        ));
    }

    let invalid_argument =
        invalid_argument.ok_or_else(|| "invalid-argument probe did not run".to_owned())?;
    let invalid_message = invalid_argument.error.unwrap_or_default();
    println!(
        "{{\"status\":\"ok\",\"boundary\":\"rust-c-abi\",\"stage\":\"qwen3-session-lifecycle\",\"device\":\"{}\",\"pcm_samples\":{},\"language\":{},\"lifecycle_count\":{},\"runs_per_lifecycle\":{},\"stable_errors\":{{\"invalid_argument\":{{\"status\":{},\"message\":{}}}}},\"sessions\":[{}]}}",
        device,
        audio.len(),
        json_string(language.to_str().unwrap_or_default()),
        lifecycle_count,
        runs_per_lifecycle,
        invalid_argument.status,
        json_string(&invalid_message),
        sessions.join(",")
    );
    Ok(())
}

fn cancel(
    model_directory: &str,
    pcm_path: &str,
    language: &str,
    device: &str,
) -> Result<(), String> {
    let model_directory = CString::new(model_directory)
        .map_err(|_| "model path contains an embedded NUL byte".to_owned())?;
    let language =
        CString::new(language).map_err(|_| "language contains an embedded NUL byte".to_owned())?;
    let device_kind = parse_device(device)?;
    let audio = read_audio(pcm_path)?;
    let session = create_session(&model_directory, device_kind)?;

    let barrier = Arc::new(Barrier::new(2));
    let worker_barrier = Arc::clone(&barrier);
    let handle_address = session.handle as usize;
    let worker_audio = audio.clone();
    let worker_language = language.clone();
    let worker = thread::spawn(move || {
        worker_barrier.wait();
        let started = Instant::now();
        let response = session_call(
            handle_address as *mut c_void,
            worker_audio.as_ptr(),
            worker_audio.len(),
            worker_language.as_ptr(),
        );
        (started.elapsed().as_secs_f64() * 1_000.0, response)
    });

    barrier.wait();
    thread::sleep(Duration::from_millis(50));
    let busy = session_call(
        session.handle,
        audio.as_ptr(),
        audio.len(),
        language.as_ptr(),
    );
    if busy.status != 4 || busy.json.is_some() {
        unsafe { cuttledoc_qwen3_mlx_session_cancel(session.handle) };
        let _ = worker.join();
        return Err(format!(
            "busy-session probe returned status {}, expected 4",
            busy.status
        ));
    }
    let cancel_started = Instant::now();
    unsafe { cuttledoc_qwen3_mlx_session_cancel(session.handle) };
    let (worker_wall_ms, cancelled) = worker
        .join()
        .map_err(|_| "cancellation worker panicked".to_owned())?;
    let cancel_to_return_ms = cancel_started.elapsed().as_secs_f64() * 1_000.0;
    if cancelled.status != 3 || cancelled.json.is_some() {
        return Err(format!(
            "cancelled transcription returned status {}, expected 3: {}",
            cancelled.status,
            cancelled.error.as_deref().unwrap_or("no error message")
        ));
    }

    println!(
        "{{\"status\":\"ok\",\"boundary\":\"rust-c-abi\",\"stage\":\"qwen3-session-cancellation\",\"device\":\"{}\",\"pcm_samples\":{},\"language\":{},\"worker_wall_ms\":{worker_wall_ms:.6},\"cancel_to_return_ms\":{cancel_to_return_ms:.6},\"busy_probe\":{{\"status\":{},\"message\":{}}},\"cancelled_call\":{{\"status\":{},\"message\":{}}}}}",
        device,
        audio.len(),
        json_string(language.to_str().unwrap_or_default()),
        busy.status,
        json_string(busy.error.as_deref().unwrap_or_default()),
        cancelled.status,
        json_string(cancelled.error.as_deref().unwrap_or_default())
    );
    Ok(())
}

fn create_session(model_directory: &CString, device_kind: i32) -> Result<MlxSession, String> {
    let mut status = 0;
    let mut error = ptr::null_mut();
    let handle = unsafe {
        cuttledoc_qwen3_mlx_session_create(
            model_directory.as_ptr(),
            device_kind,
            &mut status,
            &mut error,
        )
    };
    let error = take_string(error);
    if handle.is_null() || status != 0 {
        return Err(error.unwrap_or_else(|| {
            format!("MLX session creation returned status {status} without a message")
        }));
    }
    Ok(MlxSession { handle })
}

fn session_call(
    handle: *mut c_void,
    audio: *const f32,
    audio_len: usize,
    language: *const c_char,
) -> SessionResponse {
    let mut json = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        cuttledoc_qwen3_mlx_session_transcribe(
            handle, audio, audio_len, language, &mut json, &mut error,
        )
    };
    SessionResponse {
        status,
        json: take_string(json),
        error: take_string(error),
    }
}

fn read_audio(pcm_path: &str) -> Result<Vec<f32>, String> {
    let pcm_bytes =
        fs::read(pcm_path).map_err(|error| format!("could not read PCM fixture: {error}"))?;
    if pcm_bytes.is_empty() || pcm_bytes.len() % 4 != 0 {
        return Err("PCM fixture must contain non-empty little-endian float32".to_owned());
    }
    Ok(pcm_bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte PCM chunk")))
        .collect())
}

fn parse_device(device: &str) -> Result<i32, String> {
    match device {
        "cpu" => Ok(0),
        "gpu" => Ok(1),
        _ => Err("device must be cpu or gpu".to_owned()),
    }
}

fn parse_positive(value: &str, name: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("{name} must be a positive integer: {error}"))?;
    if parsed == 0 {
        return Err(format!("{name} must be greater than zero"));
    }
    Ok(parsed)
}

fn json_string(value: &str) -> String {
    let mut result = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\u{08}' => result.push_str("\\b"),
            '\u{0c}' => result.push_str("\\f"),
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

fn usage() -> String {
    "usage:\n  cuttledoc-qwen3-mlx-inspect MODEL_DIR\n  cuttledoc-qwen3-mlx-inspect frontend MODEL_DIR PCM_F32LE cpu|gpu\n  cuttledoc-qwen3-mlx-inspect encoder MODEL_DIR PCM_F32LE cpu|gpu\n  cuttledoc-qwen3-mlx-inspect prompt MODEL_DIR PCM_F32LE LANGUAGE cpu|gpu\n  cuttledoc-qwen3-mlx-inspect decoder MODEL_DIR PCM_F32LE LANGUAGE cpu|gpu\n  cuttledoc-qwen3-mlx-inspect transcribe MODEL_DIR PCM_F32LE LANGUAGE cpu|gpu\n  cuttledoc-qwen3-mlx-inspect lifecycle MODEL_DIR PCM_F32LE LANGUAGE cpu|gpu LIFECYCLES RUNS\n  cuttledoc-qwen3-mlx-inspect cancel MODEL_DIR PCM_F32LE LANGUAGE cpu|gpu".to_owned()
}

fn take_string(value: *mut c_char) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let result = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe { cuttledoc_qwen3_mlx_free_string(value) };
    Some(result)
}
