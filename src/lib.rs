
use faer_ext::IntoFaer;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::*;
mod linalg;
use linalg::lstsq_solver1;
mod changepoint_detection;
mod feature_extraction;
mod preprocessing;

#[pymodule]
#[pyo3(name = "_functime_rust")]
fn _functime_rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {

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
