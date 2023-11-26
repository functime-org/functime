use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
struct CusumKwargs {
    threshold: f64,
    drift: f64,
    warmup_period: usize,
}

/// This function implements a Cumulative Sum (CUSUM) filter, a quality-control method designed to
/// detect a shift in the mean value of a measured quantity away from a target value.
/// The filter is applied to a series of inputs and returns a series of events where a change has been detected.
///
/// # Arguments
///
/// * `inputs` - A slice of Series to which the CUSUM filter will be applied.
/// * `kwargs` - A struct of CusumKwargs which includes:
///     * `threshold` - The change threshold for the filter.
///     * `drift` - The expected drift in the input values.
///     * `warmup_period` - The initial period during which the filter parameters are calculated.
///
/// # Returns
///
/// * `PolarsResult<Series>` - A series of events where a change has been detected.
///
/// # Errors
///
/// This function will return an error if the inputs cannot be converted to a f64 series.
///
#[polars_expr(output_type=Int32)]
pub fn cusum(inputs: &[Series], kwargs: CusumKwargs) -> PolarsResult<Series> {
    let values = inputs[0].f64()?;

    let mut events: Vec<i32> = Vec::with_capacity(values.len());
    let mut s_pos = 0.0;
    let mut s_neg = 0.0;
    let mut t = 0;
    let mut mu_x = 0.0;
    let mut sigma_x = 0.0;
    let mut obs: Vec<f64> = Vec::new();
    let warmup_period = kwargs.warmup_period;

    for value in values.into_iter() {
        let warming_up = t < warmup_period;
        let warmup_end: bool = t == warmup_period;
        match (warming_up, warmup_end) {
            (true, _) => {
                if let Some(value) = value {
                    obs.push(value);
                }
                events.push(0);
                t += 1;
                continue;
            }
            (false, true) => {
                mu_x = obs.iter().sum::<f64>() / obs.len() as f64;
                sigma_x = obs.iter().map(|x| (x - mu_x).powi(2)).sum::<f64>() / obs.len() as f64;
                sigma_x = sigma_x.sqrt();
                t += 1;
            }
            (false, false) => {}
        }

        match value {
            Some(value) => {
                let v = (value - mu_x) / sigma_x;
                s_pos = (s_pos + v - kwargs.drift).max(0.0);
                s_neg = (s_neg + v + kwargs.drift).min(0.0);
                if s_pos > kwargs.threshold {
                    events.push(1);
                    s_pos = 0.0;
                    s_neg = 0.0;
                    t = 0;
                    obs.clear();
                } else if s_neg < -kwargs.threshold {
                    events.push(1);
                    s_neg = 0.0;
                    s_pos = 0.0;
                    t = 0;
                    obs.clear();
                } else {
                    events.push(0);
                }
            }
            _ => {
                events.push(0);
            }
        }
    }
    Ok(Series::from_vec("events", events))
}
