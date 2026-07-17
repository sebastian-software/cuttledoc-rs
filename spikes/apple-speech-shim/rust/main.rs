use std::{
    env,
    ffi::{c_void, CStr, CString},
    fs,
    os::raw::c_char,
    ptr,
    sync::Mutex,
    thread,
    time::{Duration, Instant},
};

type UpdateCallback = extern "C" fn(*mut c_void, *const c_char);

unsafe extern "C" {
    fn cuttledoc_speech_locale_inventory(output: *mut *mut c_char) -> i32;
    fn cuttledoc_speech_session_create(
        locale: *const c_char,
        sample_rate: u32,
        callback: UpdateCallback,
        callback_context: *mut c_void,
        metadata: *mut *mut c_char,
        error_output: *mut *mut c_char,
    ) -> *mut c_void;
    fn cuttledoc_speech_session_push_pcm_f32(
        handle: *mut c_void,
        samples: *const f32,
        sample_count: u32,
    ) -> i32;
    fn cuttledoc_speech_session_finish(
        handle: *mut c_void,
        summary: *mut *mut c_char,
        error_output: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_speech_session_cancel(handle: *mut c_void);
    fn cuttledoc_speech_session_destroy(handle: *mut c_void);
    fn cuttledoc_speech_free_string(value: *mut c_char);
}

extern "C" fn receive_update(context: *mut c_void, update: *const c_char) {
    if context.is_null() || update.is_null() {
        return;
    }
    let sink = unsafe { &*(context.cast::<Mutex<Vec<String>>>()) };
    let update = unsafe { CStr::from_ptr(update) }
        .to_string_lossy()
        .into_owned();
    sink.lock().expect("update sink poisoned").push(update);
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = env::args().skip(1);
    let pcm_path = arguments.next().ok_or_else(|| {
        "usage: cuttledoc-speech-spike /path/to/f32le-pcm [--locale xx-YY] [--cancel]"
            .to_owned()
    })?;
    let mut cancel_probe = false;
    let mut locale = "en-US".to_owned();
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--cancel" => cancel_probe = true,
            "--locale" => {
                locale = arguments
                    .next()
                    .ok_or_else(|| "--locale requires a value".to_owned())?;
            }
            _ => return Err(format!("unknown argument: {argument}")),
        }
    }

    let inventory = call_string(|output| unsafe {
        cuttledoc_speech_locale_inventory(output)
    })?;
    println!("LOCALE_INVENTORY {inventory}");

    let bytes = fs::read(&pcm_path)
        .map_err(|error| format!("could not read {pcm_path}: {error}"))?;
    if bytes.len() % 4 != 0 {
        return Err("PCM byte count is not divisible by four".to_owned());
    }
    let samples: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();

    let sink = Box::new(Mutex::new(Vec::<String>::new()));
    let sink_pointer = Box::into_raw(sink);
    let locale = CString::new(locale).map_err(|_| "locale contains NUL".to_owned())?;
    let mut metadata = ptr::null_mut();
    let mut create_error = ptr::null_mut();
    let create_started = Instant::now();
    let handle = unsafe {
        cuttledoc_speech_session_create(
            locale.as_ptr(),
            16_000,
            receive_update,
            sink_pointer.cast(),
            &mut metadata,
            &mut create_error,
        )
    };
    if handle.is_null() {
        let error = take_optional_string(create_error)
            .unwrap_or_else(|| "session creation failed without an error".to_owned());
        unsafe {
            drop(Box::from_raw(sink_pointer));
        }
        return Err(error);
    }
    println!("CREATE_MS {:.6}", create_started.elapsed().as_secs_f64() * 1_000.0);
    println!(
        "SESSION_METADATA {}",
        take_optional_string(metadata).unwrap_or_else(|| "{}".to_owned())
    );

    let operation = if cancel_probe {
        match push_chunk(handle, &samples[..samples.len().min(3_200)]) {
            Ok(()) => {
                unsafe {
                    cuttledoc_speech_session_cancel(handle);
                }
                println!("CANCELLED true");
                Ok(())
            }
            Err(error) => Err(error),
        }
    } else {
        let push_result = samples
            .chunks(3_200)
            .try_for_each(|chunk| push_chunk(handle, chunk));
        match push_result {
            Err(error) => Err(error),
            Ok(()) => {
                let mut summary = ptr::null_mut();
                let mut finish_error = ptr::null_mut();
                let status = unsafe {
                    cuttledoc_speech_session_finish(
                        handle,
                        &mut summary,
                        &mut finish_error,
                    )
                };
                if status != 0 {
                    Err(take_optional_string(finish_error).unwrap_or_else(|| {
                        format!("session finish failed with status {status}")
                    }))
                } else {
                    println!(
                        "SESSION_SUMMARY {}",
                        take_optional_string(summary)
                            .unwrap_or_else(|| "{}".to_owned())
                    );
                    Ok(())
                }
            }
        }
    };

    unsafe {
        cuttledoc_speech_session_destroy(handle);
    }
    let inventory_after = call_string(|output| unsafe {
        cuttledoc_speech_locale_inventory(output)
    })?;
    println!("LOCALE_INVENTORY_AFTER {inventory_after}");
    let updates = unsafe { Box::from_raw(sink_pointer) }
        .into_inner()
        .map_err(|_| "update sink poisoned".to_owned())?;
    for update in &updates {
        println!("UPDATE {update}");
    }

    operation?;
    if !cancel_probe {
        if updates.is_empty() {
            return Err("SpeechTranscriber emitted no updates".to_owned());
        }
        if !updates
            .iter()
            .any(|update| update.contains("\"stability\":\"final\""))
        {
            return Err("SpeechTranscriber emitted no final update".to_owned());
        }
        if !updates
            .iter()
            .any(|update| update.contains("\"start_ms\":"))
        {
            return Err("SpeechTranscriber emitted no audio time ranges".to_owned());
        }
        if !updates
            .iter()
            .any(|update| update.contains("\"confidence\":"))
        {
            return Err("SpeechTranscriber emitted no confidence attributes".to_owned());
        }
    }
    Ok(())
}

fn push_chunk(handle: *mut c_void, chunk: &[f32]) -> Result<(), String> {
    loop {
        let status = unsafe {
            cuttledoc_speech_session_push_pcm_f32(
                handle,
                chunk.as_ptr(),
                u32::try_from(chunk.len()).expect("chunk length fits u32"),
            )
        };
        match status {
            0 => return Ok(()),
            4 => thread::sleep(Duration::from_millis(2)),
            _ => return Err(format!("PCM push failed with status {status}")),
        }
    }
}

fn call_string(
    operation: impl FnOnce(*mut *mut c_char) -> i32,
) -> Result<String, String> {
    let mut output = ptr::null_mut();
    let status = operation(&mut output);
    let message = take_optional_string(output)
        .unwrap_or_else(|| format!("operation returned status {status} without output"));
    if status == 0 {
        Ok(message)
    } else {
        Err(message)
    }
}

fn take_optional_string(value: *mut c_char) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let message = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        cuttledoc_speech_free_string(value);
    }
    Some(message)
}
