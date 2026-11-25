use faer::prelude::*;
use faer::solvers::SolverCore;
use faer::{MatRef, Side};
use faer_ext::{IntoFaer, IntoNdarray};
use ndarray::Array2;

#[inline]
pub fn lstsq_solver1(x: MatRef<f64>, y: MatRef<f64>) -> Array2<f64> {
    // Fast-path normal equations only when full rank and well-conditioned;
    // otherwise Cholesky panics or loses precision on constant/rank-deficient inputs.
    if x.nrows() >= x.ncols() {
        let xt = x.transpose();
        let xtx = xt * x;
        if let Ok(cholesky) = xtx.cholesky(Side::Lower) {
            let l = cholesky.compute_l();
            let mut min_diag = f64::INFINITY;
            let mut max_diag = 0.0_f64;
            for i in 0..l.nrows() {
                let d = l.read(i, i).abs();
                if d < min_diag {
                    min_diag = d;
                }
                if d > max_diag {
                    max_diag = d;
                }
            }
            if max_diag > 0.0 && min_diag > max_diag * 1e-7 {
                let xtx_inv = cholesky.inverse();
                let beta = xtx_inv * xt * y;
                return beta.as_ref().into_ndarray().to_owned();
            }
        }
    }
    let beta = x.thin_svd().pseudoinverse() * y;
    beta.as_ref().into_ndarray().to_owned()
}
