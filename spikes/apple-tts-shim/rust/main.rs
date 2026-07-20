use std::{
    env,
    ffi::{c_void, CStr, CString},
    fs,
    os::raw::c_char,
    ptr, thread,
    time::{Duration, Instant},
};

unsafe extern "C" {
    fn cuttledoc_tts_voice_inventory(
        locale: *const c_char,
        output: *mut *mut c_char,
        error_output: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_tts_session_create(
        locale: *const c_char,
        voice_identifier: *const c_char,
        metadata: *mut *mut c_char,
        error_output: *mut *mut c_char,
    ) -> *mut c_void;
    fn cuttledoc_tts_session_synthesize(
        handle: *mut c_void,
        text: *const c_char,
        samples: *mut *mut f32,
        sample_count: *mut u64,
        sample_rate_hz: *mut u32,
        summary: *mut *mut c_char,
        error_output: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_tts_session_cancel(handle: *mut c_void);
    fn cuttledoc_tts_session_destroy(handle: *mut c_void);
    fn cuttledoc_tts_free_audio(samples: *mut f32);
    fn cuttledoc_tts_free_string(value: *mut c_char);
}

struct Session {
    handle: *mut c_void,
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            cuttledoc_tts_session_destroy(self.handle);
        }
    }
}

struct SynthesisCall {
    status: i32,
    samples: Vec<f32>,
    sample_rate_hz: u32,
    summary: Option<String>,
    error: Option<String>,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    match arguments.as_slice() {
        [command, locale] if command == "inventory" => inventory(locale),
        [command, text_path, output_path, rest @ ..] if command == "synthesize" => {
            let (locale, voice) = parse_options(rest)?;
            synthesize(text_path, output_path, &locale, voice.as_deref())
        }
        [command, text_path, rest @ ..] if command == "cancel" => {
            let (locale, voice) = parse_options(rest)?;
            cancel(text_path, &locale, voice.as_deref())
        }
        _ => Err(usage()),
    }
}

fn parse_options(arguments: &[String]) -> Result<(String, Option<String>), String> {
    let mut locale = "de-DE".to_owned();
    let mut voice = None;
    let mut index = 0;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--locale" => {
                index += 1;
                locale = arguments
                    .get(index)
                    .ok_or_else(|| "--locale requires a value".to_owned())?
                    .clone();
            }
            "--voice" => {
                index += 1;
                voice = Some(
                    arguments
                        .get(index)
                        .ok_or_else(|| "--voice requires a value".to_owned())?
                        .clone(),
                );
            }
            argument => return Err(format!("unknown argument: {argument}")),
        }
        index += 1;
    }
    Ok((locale, voice))
}

fn inventory(locale: &str) -> Result<(), String> {
    let locale = CString::new(locale).map_err(|_| "locale contains NUL".to_owned())?;
    let mut output = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe { cuttledoc_tts_voice_inventory(locale.as_ptr(), &mut output, &mut error) };
    if status != 0 {
        return Err(take_optional_string(error)
            .unwrap_or_else(|| format!("voice inventory failed with status {status}")));
    }
    println!(
        "{}",
        take_optional_string(output).unwrap_or_else(|| "{}".to_owned())
    );
    Ok(())
}

fn create_session(locale: &str, voice: Option<&str>) -> Result<(Session, String), String> {
    let locale = CString::new(locale).map_err(|_| "locale contains NUL".to_owned())?;
    let voice = voice
        .map(CString::new)
        .transpose()
        .map_err(|_| "voice identifier contains NUL".to_owned())?;
    let mut metadata = ptr::null_mut();
    let mut error = ptr::null_mut();
    let handle = unsafe {
        cuttledoc_tts_session_create(
            locale.as_ptr(),
            voice.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            &mut metadata,
            &mut error,
        )
    };
    if handle.is_null() {
        return Err(take_optional_string(error)
            .unwrap_or_else(|| "session creation failed without an error".to_owned()));
    }
    Ok((
        Session { handle },
        take_optional_string(metadata).unwrap_or_else(|| "{}".to_owned()),
    ))
}

fn synthesize(
    text_path: &str,
    output_path: &str,
    locale: &str,
    voice: Option<&str>,
) -> Result<(), String> {
    let text = fs::read_to_string(text_path)
        .map_err(|error| format!("could not read {text_path}: {error}"))?;
    let create_started = Instant::now();
    let (session, metadata) = create_session(locale, voice)?;
    println!(
        "CREATE_MS {:.6}",
        create_started.elapsed().as_secs_f64() * 1_000.0
    );
    println!("SESSION_METADATA {metadata}");
    let call = call_synthesize(session.handle, &text)?;
    if call.status != 0 {
        return Err(call
            .error
            .unwrap_or_else(|| format!("synthesis failed with status {}", call.status)));
    }
    let bytes: Vec<u8> = call
        .samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect();
    fs::write(output_path, bytes)
        .map_err(|error| format!("could not write {output_path}: {error}"))?;
    println!(
        "SYNTHESIS_SUMMARY {}",
        call.summary.unwrap_or_else(|| "{}".to_owned())
    );
    println!("OUTPUT_PATH {output_path}");
    println!("OUTPUT_SAMPLE_RATE_HZ {}", call.sample_rate_hz);
    println!("OUTPUT_SAMPLE_COUNT {}", call.samples.len());
    drop(session);
    Ok(())
}

fn cancel(text_path: &str, locale: &str, voice: Option<&str>) -> Result<(), String> {
    let text = fs::read_to_string(text_path)
        .map_err(|error| format!("could not read {text_path}: {error}"))?
        .repeat(8);
    let (session, metadata) = create_session(locale, voice)?;
    println!("SESSION_METADATA {metadata}");
    let handle = session.handle as usize;
    let worker = thread::spawn(move || call_synthesize(handle as *mut c_void, &text));
    thread::sleep(Duration::from_millis(10));

    let busy = call_synthesize(session.handle, "busy probe")?;
    if busy.status != 4 {
        return Err(format!(
            "concurrent synthesis returned {}, expected busy status 4",
            busy.status
        ));
    }
    let cancel_started = Instant::now();
    unsafe {
        cuttledoc_tts_session_cancel(session.handle);
    }
    let cancelled = worker
        .join()
        .map_err(|_| "cancellation worker panicked".to_owned())??;
    let cancel_to_return_ms = cancel_started.elapsed().as_secs_f64() * 1_000.0;
    if cancelled.status != 3 || !cancelled.samples.is_empty() {
        return Err(format!(
            "cancelled synthesis returned status {} and {} samples",
            cancelled.status,
            cancelled.samples.len()
        ));
    }
    println!(
        "CANCELLATION {{\"status\":\"ok\",\"busy_status\":{},\"cancelled_status\":{},\"cancel_to_return_ms\":{cancel_to_return_ms:.6},\"summary\":{},\"message\":{}}}",
        busy.status,
        cancelled.status,
        cancelled.summary.as_deref().unwrap_or("null"),
        json_string(cancelled.error.as_deref().unwrap_or_default()),
    );
    drop(session);
    Ok(())
}

fn call_synthesize(handle: *mut c_void, text: &str) -> Result<SynthesisCall, String> {
    let text = CString::new(text).map_err(|_| "text contains NUL".to_owned())?;
    let mut samples = ptr::null_mut();
    let mut sample_count = 0_u64;
    let mut sample_rate_hz = 0_u32;
    let mut summary = ptr::null_mut();
    let mut error = ptr::null_mut();
    let status = unsafe {
        cuttledoc_tts_session_synthesize(
            handle,
            text.as_ptr(),
            &mut samples,
            &mut sample_count,
            &mut sample_rate_hz,
            &mut summary,
            &mut error,
        )
    };
    let copied = if samples.is_null() {
        Vec::new()
    } else {
        let length = usize::try_from(sample_count)
            .map_err(|_| "sample count does not fit usize".to_owned())?;
        let copied = unsafe { std::slice::from_raw_parts(samples, length) }.to_vec();
        unsafe {
            cuttledoc_tts_free_audio(samples);
        }
        copied
    };
    Ok(SynthesisCall {
        status,
        samples: copied,
        sample_rate_hz,
        summary: take_optional_string(summary),
        error: take_optional_string(error),
    })
}

fn take_optional_string(value: *mut c_char) -> Option<String> {
    if value.is_null() {
        return None;
    }
    let message = unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        cuttledoc_tts_free_string(value);
    }
    Some(message)
}

fn json_string(value: &str) -> String {
    let mut output = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => output.push(character),
        }
    }
    output.push('"');
    output
}

fn usage() -> String {
    "usage:\n  cuttledoc-apple-tts-spike inventory LOCALE\n  \
     cuttledoc-apple-tts-spike synthesize TEXT_FILE OUTPUT_F32LE \
     [--locale xx-YY] [--voice IDENTIFIER]\n  \
     cuttledoc-apple-tts-spike cancel TEXT_FILE \
     [--locale xx-YY] [--voice IDENTIFIER]"
        .to_owned()
}
