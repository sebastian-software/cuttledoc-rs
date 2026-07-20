use std::{
    env,
    ffi::{CStr, CString, c_char},
    ptr,
};

unsafe extern "C" {
    fn cuttledoc_qwen3_mlx_inspect_model(
        model_directory: *const c_char,
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
    let mut arguments = env::args().skip(1);
    let model_directory = arguments.next().ok_or_else(usage)?;
    if arguments.next().is_some() {
        return Err(usage());
    }
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

fn usage() -> String {
    "usage: cuttledoc-qwen3-mlx-inspect MODEL_DIR".to_owned()
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
