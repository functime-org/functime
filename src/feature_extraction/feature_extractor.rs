use polars_core::prelude::*;
use hashbrown::HashSet;
use pyo3_polars::{derive::polars_expr, export::polars_core::{series::Series, prelude::{*}}};
//use pyo3::prelude::*;

#[polars_expr(output_type=UInt32)]
fn pl_lempel_ziv_complexity(inputs: &[Series]) -> PolarsResult<Series>  {
    
    let input: &Series = &inputs[0];
    let name = input.name();
    let input = input.bool()?;
    let bits: Vec<bool> = input.into_iter().map(
        |op_b| op_b.unwrap_or(false)
    ).collect();

    let mut ind:usize = 0;
    let mut inc:usize = 1;

    let mut sub_strings:HashSet<&[bool]> = HashSet::new(); 
    while ind + inc <= bits.len() {
        let subseq: &[bool] = &bits[ind..ind+inc];
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




// // Test this when Faer updates its Polars interop

//     let input: &Series = &inputs[0];
//     let n_lag: &u32 = &inputs[1].u32()?.get(0).unwrap();
//     let name: &str = input.name();

//     let todf: Result<DataFrame, PolarsError> = df!(name => input);
//     match todf {
//         Ok(df) => {
//             let length: u32 = inputs.len() as u32 - n_lag;
//             let mut shifts:Vec<Expr> = (1..(*n_lag + 1)).map(|i| {
//                 col(name).slice(n_lag - i, length).alias(i.to_string().as_ref())
//             }).collect();
//             shifts.push(lit(1.0));
//             // Construct X
//             let df_lazy_x: LazyFrame = df.lazy().select(shifts);
//             // Construct Y
//             let df_lazy_y: LazyFrame = df.lazy().select(
//                 [col(name).tail(Some(length as usize)).alias("target")]
//             );
//             let mat_x = polars_to_faer_f64(df_lazy_x);
//             let mat_y = polars_to_faer_f64(df_lazy_y);
//             let coeffs = match (mat_x, mat_y) {
//                 // Use lstsq_solver1 because it is better for matrix that has nrows >>> ncols
//                 (Ok(x), Ok(y)) => {
//                     use super::lstsq_solver1;
//                     use ndarray::Array;
//                     lstsq_solver1(x, y)
//                 },
//                 _ => {
//                     return PolarsError::ComputeError("Cannot convert autoregressive matrix to Faer matrix.")
//                 } 
//             };
//             // Coeffs is a 2d array because Faer returns a matrix (the vector as a matrix)
//             // coeffs.into_iter() traverses every element in order.
//             let output: Series = Series::from_iter(coeffs.into_iter());
//             Ok(output)
//         }
//         , Err(e) => {
//             return PolarsResult::Err(e)
//         }
//     }
// }