use std::{
    env,
    ffi::{CStr, CString, c_char},
    fs, ptr,
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

fn usage() -> String {
    "usage:\n  cuttledoc-qwen3-mlx-inspect MODEL_DIR\n  cuttledoc-qwen3-mlx-inspect frontend MODEL_DIR PCM_F32LE cpu|gpu\n  cuttledoc-qwen3-mlx-inspect encoder MODEL_DIR PCM_F32LE cpu|gpu\n  cuttledoc-qwen3-mlx-inspect prompt MODEL_DIR PCM_F32LE LANGUAGE cpu|gpu\n  cuttledoc-qwen3-mlx-inspect decoder MODEL_DIR PCM_F32LE LANGUAGE cpu|gpu\n  cuttledoc-qwen3-mlx-inspect transcribe MODEL_DIR PCM_F32LE LANGUAGE cpu|gpu".to_owned()
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
