use faer::IntoFaer;
use numpy::{PyReadonlyArray2, PyArray2, ToPyArray};
use pyo3::prelude::*;
mod feature_extraction;
pub mod linalg;
use linalg::lstsq_solver1;

#[pymodule]
fn _functime_rust(_py: Python, m: &PyModule) -> PyResult<()> {

    // Normal Rust function interop
    // m.add_function(wrap_pyfunction!(rs_lempel_ziv_complexity, m)?)?;

    // Functions that Requires NumPy Interop
    #[pyfn(m)]
    fn rs_faer_lstsq1<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<f64> {
        let x_ = x.as_array();
        let y_ = y.as_array();
        let beta = lstsq_solver1(x_.into_faer(), y_.into_faer());
        beta.to_pyarray(py)
    }
    Ok(())
}
