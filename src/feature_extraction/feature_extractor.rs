use hashbrown::HashSet;
use polars_core::prelude::*;
use pyo3_polars::{
    derive::polars_expr,
    export::polars_core::{prelude::*, series::Series},
};
//use pyo3::prelude::*;

#[polars_expr(output_type=UInt32)]
fn pl_lempel_ziv_complexity(inputs: &[Series]) -> PolarsResult<Series> {
    let input: &Series = &inputs[0];
    let name = input.name();
    let input = input.bool()?;
    let bits: Vec<bool> = input
        .into_iter()
        .map(|op_b| op_b.unwrap_or(false))
        .collect();

    let mut ind: usize = 0;
    let mut inc: usize = 1;

    let mut sub_strings: HashSet<&[bool]> = HashSet::new();
    while ind + inc <= bits.len() {
        let subseq: &[bool] = &bits[ind..ind + inc];
        if sub_strings.contains(subseq) {
            inc += 1;
        } else {
            sub_strings.insert(subseq);
            ind += inc;
            inc = 1;
        }
    }
    let c = sub_strings.len();
    Ok(Series::new(name, [c as u32]))
}
