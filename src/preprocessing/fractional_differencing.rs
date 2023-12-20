use ndarray::prelude::*;
use ndarray::Array1;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

fn get_ffd_weights(d: f64, threshold: Option<f64>, window_size: Option<usize>) -> Array1<f64> {
    let mut weights: Vec<f64> = Vec::from([1.0]);
    let mut k = 1;
    loop {
        let weight = -weights[k - 1] * (d - k as f64 + 1.0) / k as f64;
        if weight.abs() < threshold.unwrap_or(0.0) || k > window_size.unwrap_or(1000) {
            break;
        }
        weights.push(weight);
        k += 1;
    }
    // reverse the weights
    weights.reverse();
    Array1::from(weights)
}

#[derive(Deserialize)]
struct FracDiffKwargs {
    d: f64,
    min_weight: Option<f64>,
    window_size: Option<usize>,
}
#[polars_expr(output_type=Float64)]
pub fn frac_diff(inputs: &[Series], kwargs: FracDiffKwargs) -> PolarsResult<Series> {
    let weights = get_ffd_weights(kwargs.d, kwargs.min_weight, kwargs.window_size);
    let values = inputs[0].f64().unwrap().to_ndarray()?;
    let mut output_builder = PrimitiveChunkedBuilder::<Float64Type>::new("frac_diff", values.len());
    let width = weights.len() - 1;
    for i in 0..values.len() {
        if i < width {
            output_builder.append_null();
        } else {
            let value = weights.t().dot(&values.slice(s![i - width..=i]));
            output_builder.append_value(value);
        }
    }
    Ok(output_builder.finish().into_series())
}
