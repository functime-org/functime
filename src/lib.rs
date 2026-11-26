mod changepoint_detection;
mod feature_extraction;

use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "_functime_rust")]
fn _functime_rust(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
