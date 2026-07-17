use napi::{Error, Result, Status};
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
