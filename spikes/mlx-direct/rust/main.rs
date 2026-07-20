use std::{
    env,
    ffi::{CStr, CString, c_char, c_void},
    fs, ptr,
    time::Instant,
};

unsafe extern "C" {
    fn cuttledoc_mlx_whisper_create(
        model_directory: *const c_char,
        device_kind: i32,
        error_out: *mut *mut c_char,
    ) -> *mut c_void;
    fn cuttledoc_mlx_whisper_describe(
        handle: *mut c_void,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_mlx_whisper_transcribe(
        handle: *mut c_void,
        audio: *const f32,
        audio_len: usize,
        json_out: *mut *mut c_char,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_mlx_whisper_destroy(handle: *mut c_void);
    fn cuttledoc_mlx_free_string(value: *mut c_char);
}

struct Session(*mut c_void);

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { cuttledoc_mlx_whisper_destroy(self.0) };
    }
}

struct SessionResult {
    load_ms: f64,
    load_description: String,
    runs: Vec<(f64, String)>,
    destroy_ms: f64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let model_directory = arguments.next().ok_or_else(usage)?;
    let pcm_path = arguments.next().ok_or_else(usage)?;
    let device_name = arguments.next().ok_or_else(usage)?;
    let device_kind = match device_name.as_str() {
        "cpu" => 0,
        "gpu" => 1,
        _ => return Err("device must be cpu or gpu".to_owned()),
    };
    let lifecycle_count = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| format!("invalid lifecycle count: {error}"))?
        .unwrap_or(3);
    let runs_per_lifecycle = arguments
        .next()
        .map(|value| value.parse::<usize>())
        .transpose()
        .map_err(|error| format!("invalid runs-per-lifecycle count: {error}"))?
        .unwrap_or(2);
    if arguments.next().is_some() || lifecycle_count == 0 || runs_per_lifecycle == 0 {
        return Err(usage());
    }

    let bytes =
        fs::read(&pcm_path).map_err(|error| format!("could not read {pcm_path}: {error}"))?;
    if bytes.is_empty() || bytes.len() % 4 != 0 {
        return Err("PCM fixture must contain a non-empty number of f32le samples".to_owned());
    }
    let audio = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();

    let model_directory = CString::new(model_directory)
        .map_err(|_| "model path contains an embedded NUL byte".to_owned())?;
    let mut sessions = Vec::with_capacity(lifecycle_count);
    for _ in 0..lifecycle_count {
        let load_started = Instant::now();
        let mut session = create_session(&model_directory, device_kind)?;
        let load_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
        let load_description = call_json(|json, error| unsafe {
            cuttledoc_mlx_whisper_describe(session.0, json, error)
        })?;

        let mut runs = Vec::with_capacity(runs_per_lifecycle);
        for _ in 0..runs_per_lifecycle {
            let started = Instant::now();
            let result = call_json(|json, error| unsafe {
                cuttledoc_mlx_whisper_transcribe(
                    session.0,
                    audio.as_ptr(),
                    audio.len(),
                    json,
                    error,
                )
            })?;
            runs.push((started.elapsed().as_secs_f64() * 1_000.0, result));
        }

        let destroy_started = Instant::now();
        unsafe { cuttledoc_mlx_whisper_destroy(session.0) };
        session.0 = ptr::null_mut();
        let destroy_ms = destroy_started.elapsed().as_secs_f64() * 1_000.0;
        sessions.push(SessionResult {
            load_ms,
            load_description,
            runs,
            destroy_ms,
        });
    }

    print!(
        "{{\"device\":\"{device_name}\",\"pcm_samples\":{},\"lifecycle_count\":{lifecycle_count},\"runs_per_lifecycle\":{runs_per_lifecycle},\"sessions\":[",
        audio.len()
    );
    for (session_index, session) in sessions.iter().enumerate() {
        if session_index != 0 {
            print!(",");
        }
        print!(
            "{{\"load_wall_ms\":{:.3},\"load\":{},\"runs\":[",
            session.load_ms, session.load_description
        );
        for (run_index, (wall_ms, result)) in session.runs.iter().enumerate() {
            if run_index != 0 {
                print!(",");
            }
            print!("{{\"wall_ms\":{wall_ms:.3},\"transcription\":{result}}}");
        }
        print!("],\"destroy_wall_ms\":{:.3}}}", session.destroy_ms);
    }
    println!("]}}");
    Ok(())
}

fn usage() -> String {
    "usage: cuttledoc-mlx-whisper MODEL_DIR PCM_F32LE cpu|gpu [LIFECYCLES] [RUNS_PER_LIFECYCLE]"
        .to_owned()
}

fn create_session(model_directory: &CString, device_kind: i32) -> Result<Session, String> {
    let mut error = ptr::null_mut();
    let handle =
        unsafe { cuttledoc_mlx_whisper_create(model_directory.as_ptr(), device_kind, &mut error) };
    if !handle.is_null() {
        return Ok(Session(handle));
    }
    Err(take_string(error).unwrap_or_else(|| "MLX shim returned no handle and no error".to_owned()))
}

fn call_json(
    callback: impl FnOnce(*mut *mut c_char, *mut *mut c_char) -> i32,
) -> Result<String, String> {
    let mut json = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = callback(&mut json, &mut error);
    if status == 0 {
        return take_string(json)
            .ok_or_else(|| "MLX shim returned success without JSON".to_owned());
    }
    if !json.is_null() {
        unsafe { cuttledoc_mlx_free_string(json) };
    }
    Err(take_string(error)
        .unwrap_or_else(|| "MLX shim returned an error without a message".to_owned()))
}

fn take_string(value: *mut c_char) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let result = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe { cuttledoc_mlx_free_string(value) };
    Some(result)
}
