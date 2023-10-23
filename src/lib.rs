use pyo3::prelude::*;

mod feature_extraction;
use feature_extraction::feature_extractor::rs_lempel_ziv_complexity;

#[pymodule]
fn _functime_rust(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(rs_lempel_ziv_complexity, m)?)?;
    Ok(())
}