use std::{env, ffi::{CStr, CString}, os::raw::c_char, ptr};

unsafe extern "C" {
    fn cuttledoc_speech_transcribe_file(
        path: *const c_char,
        output: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_speech_free_string(value: *mut c_char);
}

fn main() {
    let audio_path = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: cuttledoc-speech-spike /absolute/path/to/audio");
        std::process::exit(2);
    });
    let audio_path = CString::new(audio_path).expect("audio path must not contain a NUL byte");
    let mut output = ptr::null_mut();
    let status = unsafe { cuttledoc_speech_transcribe_file(audio_path.as_ptr(), &mut output) };

    if output.is_null() {
        eprintln!("Swift shim returned status {status} without a message");
        std::process::exit(1);
    }

    let message = unsafe { CStr::from_ptr(output) }.to_string_lossy().into_owned();
    unsafe { cuttledoc_speech_free_string(output) };

    if status == 0 {
        println!("{message}");
    } else {
        eprintln!("SpeechAnalyzer error ({status}): {message}");
        std::process::exit(status);
    }
}
