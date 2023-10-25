use pyo3::prelude::*;
use pyo3_polars::{derive::polars_expr, export::polars_core::{series::Series, prelude::{*}}};
use std::collections::HashSet;

#[pyfunction]
pub fn rs_lempel_ziv_complexity(
    s: &[u8]
) -> usize {

    let mut ind:usize = 0;
    let mut inc:usize = 1;

    let mut sub_strings: HashSet<&[u8]> = HashSet::new();
    while ind + inc <= s.len() {
        let subseq: &[u8] = &s[ind..ind+inc];
        if sub_strings.contains(subseq) {
            inc += 1;
        } else {
            sub_strings.insert(subseq);
            ind += inc;
            inc = 1;
        }
    }
    sub_strings.len()
}

// #[polars_expr(output_type=UInt32)]
// fn pl_lempel_ziv_complexity(inputs: &[Series]) -> PolarsResult<Series>  {
    
//     let input: &Series = &inputs[0];
//     let name = input.name();
//     let input = input.bool()?;
//     let bools: Vec<u8> = input.into_iter().map(
//         |op_b| (op_b.unwrap_or(false) as u8)
//     ).collect();
//     let c:usize = rs_lempel_ziv_complexity(&bools);
//     Ok(Series::new(name, [c as u32]))

// }