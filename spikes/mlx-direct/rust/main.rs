use std::{ffi::CStr, os::raw::c_char, ptr};

const AUDIO_FRAME_ELEMENTS: usize = 576;
const PROJECTION_OUTPUTS: usize = 4;

unsafe extern "C" {
    fn cuttledoc_mlx_project_audio(
        audio: *const f32,
        audio_len: usize,
        weights: *const f32,
        weights_len: usize,
        output: *mut f32,
        output_len: usize,
        device_kind: i32,
        error_out: *mut *mut c_char,
    ) -> i32;
    fn cuttledoc_mlx_free_string(value: *mut c_char);
}

fn main() {
    let audio = (0..AUDIO_FRAME_ELEMENTS)
        .map(|index| (index as f32 / 32.0).sin())
        .collect::<Vec<_>>();
    let weights = (0..AUDIO_FRAME_ELEMENTS * PROJECTION_OUTPUTS)
        .map(|index| ((index as f32 + 1.0) / 128.0).cos() / 64.0)
        .collect::<Vec<_>>();

    for (name, device) in [("cpu", 0), ("gpu", 1)] {
        let output = project(&audio, &weights, device).unwrap_or_else(|error| {
            eprintln!("{name} projection failed: {error}");
            std::process::exit(1);
        });
        println!("device={name} scores={output:?}");
    }
}

fn project(
    audio: &[f32],
    weights: &[f32],
    device: i32,
) -> Result<[f32; PROJECTION_OUTPUTS], String> {
    let mut output = [0.0; PROJECTION_OUTPUTS];
    let mut error = ptr::null_mut();
    let status = unsafe {
        cuttledoc_mlx_project_audio(
            audio.as_ptr(),
            audio.len(),
            weights.as_ptr(),
            weights.len(),
            output.as_mut_ptr(),
            output.len(),
            device,
            &mut error,
        )
    };
    if status == 0 {
        return Ok(output);
    }

    let message = if error.is_null() {
        "MLX shim returned an error without a message".to_owned()
    } else {
        let value = unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned();
        unsafe { cuttledoc_mlx_free_string(error) };
        value
    };
    Err(message)
}
