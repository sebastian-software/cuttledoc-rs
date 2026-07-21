use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

use napi::bindgen_prelude::{AbortSignal, AsyncTask, Buffer};
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi::{Env, Error, Result, Status, Task};
use napi_derive::napi;

const CONTRACT_VECTORS: &str =
    include_str!("../../../../fixtures/contracts/transcription-updates.tsv");

#[napi(object)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ContractUpdate {
    pub sequence: u32,
    pub operation: String,
    pub stability: Option<String>,
    #[napi(js_name = "affectedStartMs")]
    pub affected_start_ms: u32,
    #[napi(js_name = "affectedEndMs")]
    pub affected_end_ms: u32,
    #[napi(js_name = "segmentStartMs")]
    pub segment_start_ms: Option<u32>,
    #[napi(js_name = "segmentEndMs")]
    pub segment_end_ms: Option<u32>,
    pub text: Option<String>,
}

#[napi(object)]
pub struct PcmWorkResult {
    #[napi(js_name = "byteLength")]
    pub byte_length: u32,
    pub checksum: u32,
    #[napi(js_name = "progressSteps")]
    pub progress_steps: u32,
}

pub struct PcmWork {
    pcm: Vec<u8>,
    fail: bool,
    progress: ThreadsafeFunction<u32>,
    cancelled: Arc<AtomicBool>,
}

impl Task for PcmWork {
    type Output = PcmWorkResult;
    type JsValue = PcmWorkResult;

    fn compute(&mut self) -> Result<Self::Output> {
        let mut checksum = 0_u32;
        for progress in [25_u32, 50, 75, 100] {
            if self.cancelled.load(Ordering::Acquire) {
                return Err(Error::new(
                    Status::Cancelled,
                    "native PCM work aborted".to_owned(),
                ));
            }
            let start = self.pcm.len() * (progress as usize - 25) / 100;
            let end = self.pcm.len() * progress as usize / 100;
            checksum = self.pcm[start..end].iter().fold(checksum, |total, value| {
                total.wrapping_add(u32::from(*value))
            });
            let status = self
                .progress
                .call(Ok(progress), ThreadsafeFunctionCallMode::Blocking);
            if status != Status::Ok {
                return Err(Error::new(
                    status,
                    "could not deliver native progress update".to_owned(),
                ));
            }
            thread::sleep(Duration::from_millis(5));
        }

        if self.fail {
            return Err(Error::new(
                Status::GenericFailure,
                "simulated Rust transcription failure".to_owned(),
            ));
        }

        Ok(PcmWorkResult {
            byte_length: self.pcm.len().try_into().map_err(|_| {
                Error::new(
                    Status::InvalidArg,
                    "PCM input exceeds u32 length".to_owned(),
                )
            })?,
            checksum,
            progress_steps: 4,
        })
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output)
    }
}

#[napi(js_name = "processPcm")]
pub fn process_pcm(
    pcm: Buffer,
    fail: bool,
    progress: ThreadsafeFunction<u32>,
    signal: Option<AbortSignal>,
) -> AsyncTask<PcmWork> {
    let cancelled = Arc::new(AtomicBool::new(false));
    if let Some(abort_signal) = signal.as_ref() {
        let cancelled_on_abort = Arc::clone(&cancelled);
        abort_signal.on_abort(move || cancelled_on_abort.store(true, Ordering::Release));
    }
    let task = PcmWork {
        pcm: pcm.as_ref().to_vec(),
        fail,
        progress,
        cancelled,
    };
    AsyncTask::with_optional_signal(task, signal)
}

#[napi]
pub struct NativeContractStream {
    updates: Vec<ContractUpdate>,
    cursor: usize,
    closed: bool,
}

#[napi]
impl NativeContractStream {
    #[napi(constructor)]
    pub fn new(case_name: String) -> Result<Self> {
        let updates = parse_successful_case(&case_name)?;
        Ok(Self {
            updates,
            cursor: 0,
            closed: false,
        })
    }

    #[napi(js_name = "nextUpdate")]
    pub fn next_update(&mut self) -> Option<ContractUpdate> {
        if self.closed {
            return None;
        }

        let update = self.updates.get(self.cursor).cloned();
        if update.is_some() {
            self.cursor += 1;
        } else {
            self.closed = true;
        }
        update
    }

    #[napi]
    pub fn close(&mut self) {
        self.closed = true;
    }

    #[napi(js_name = "isClosed")]
    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

fn parse_successful_case(case_name: &str) -> Result<Vec<ContractUpdate>> {
    let mut updates = Vec::new();
    for line in CONTRACT_VECTORS.lines().skip(1) {
        let columns: Vec<_> = line.split('\t').collect();
        if columns.len() != 11 {
            return Err(invalid_fixture("contract vector must have 11 columns"));
        }
        if columns[0] != case_name || columns[10] != "-" {
            continue;
        }

        updates.push(ContractUpdate {
            sequence: parse_u32(columns[1], "sequence")?,
            operation: columns[2].to_owned(),
            stability: optional_string(columns[3]),
            affected_start_ms: parse_u32(columns[4], "affected_start_ms")?,
            affected_end_ms: parse_u32(columns[5], "affected_end_ms")?,
            segment_start_ms: optional_u32(columns[6], "segment_start_ms")?,
            segment_end_ms: optional_u32(columns[7], "segment_end_ms")?,
            text: optional_string(columns[8]),
        });
    }

    if updates.is_empty() {
        return Err(Error::new(
            Status::InvalidArg,
            format!("unknown or unsuccessful-only contract case: {case_name}"),
        ));
    }
    Ok(updates)
}

fn parse_u32(value: &str, field: &str) -> Result<u32> {
    value
        .parse()
        .map_err(|_| invalid_fixture(&format!("invalid {field}: {value}")))
}

fn optional_u32(value: &str, field: &str) -> Result<Option<u32>> {
    if value == "-" || value == "0" {
        Ok(None)
    } else {
        parse_u32(value, field).map(Some)
    }
}

fn optional_string(value: &str) -> Option<String> {
    (value != "-").then(|| value.to_owned())
}

fn invalid_fixture(reason: &str) -> Error {
    Error::new(
        Status::GenericFailure,
        format!("invalid contract fixture: {reason}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn successful_rows_are_embedded_from_the_shared_fixture() {
        let updates = parse_successful_case("volatile_to_final").unwrap();
        assert_eq!(updates.len(), 3);
        assert_eq!(updates[2].sequence, 3);
        assert_eq!(updates[2].text.as_deref(), Some("hello_world"));
    }

    #[test]
    fn unknown_case_is_rejected() {
        assert!(parse_successful_case("missing").is_err());
    }
}
