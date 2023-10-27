use ndarray::Array2;
use faer::{IntoNdarray, Side, MatRef};
use faer::prelude::*;

#[inline]
pub fn lstsq_solver1(
    x: MatRef<f64>
    , y: MatRef<f64>
) -> Array2<f64> {

    // Solver1. Use closed form solution to solve the least square
    // This is faster because xtx has small dimension. So we use the closed
    // form solution approach.
    let xt = x.transpose();
    let xtx = xt * x;
    let cholesky = xtx.cholesky(Side::Lower).unwrap(); // Can unwrap because xtx is positive semidefinite
    let xtx_inv = cholesky.inverse();
    // Solution
    let beta = xtx_inv * xt * y;
    let out = beta.as_ref().into_ndarray();
    out.to_owned()
}
