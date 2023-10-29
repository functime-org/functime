use numpy::{PyReadonlyArray2, PyArray2, ToPyArray};
use pyo3::prelude::*;
mod feature_extraction;
use feature_extraction::{
    faer_lstsq,
    feature_extractor::rs_lempel_ziv_complexity
};

#[pymodule]
fn _functime_rust(_py: Python, m: &PyModule) -> PyResult<()> {

    m.add_function(wrap_pyfunction!(rs_lempel_ziv_complexity, m)?)?;

    // Functions that Requires NumPy Interop
    #[pyfn(m)]
    fn rs_faer_lstsq<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
    ) -> &'py PyArray2<f64> {
        let x_ = x.as_array();
        let y_ = y.as_array();
        let beta = faer_lstsq(x_, y_);
        beta.to_pyarray(py)
    }
    Ok(())
}
