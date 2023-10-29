pub mod feature_extractor;
use ndarray::{Array2, ArrayView2};
use faer::{IntoFaer, IntoNdarray, Side};
use faer::prelude::*;


pub fn faer_lstsq(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
) -> Array2<f64> {

    // Solving X * beta = rhs using Faer-rs
    // This speeds things up significantly but is only suitable for m x k matrices
    // where m >> k.

    // Zero Copy
    let x_ = x.into_faer();
    let y_ = y.into_faer();
    // Solver. Use closed form solution to solve the least square
    // This is faster because xtx has small dimension. So we use the closed
    // form solution approach.
    let xt = x_.transpose();
    let xtx = xt * x_;
    let cholesky = xtx.cholesky(Side::Lower).unwrap(); // Can unwrap because xtx is positive semidefinite
    let xtx_inv = cholesky.inverse();
    // Solution
    let beta = xtx_inv * xt * y_;
    // To Ndarray (for NumPy Interop), generate output. Will Copy.
    let out = beta.as_ref().into_ndarray();
    out.to_owned()
}
