use pyo3::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};

/// Returns the current wall-clock time in nanoseconds since Unix epoch.
#[pyfunction]
pub fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
