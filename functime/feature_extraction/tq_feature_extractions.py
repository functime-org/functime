import polars as pl
from scipy.spatial import KDTree
import numpy as np
import math

# permutation_entropy: done, test?
# query_similarity_count: not yet
# range_count: done, test?
# ratio_beyond_r_sigma: done, test?
# ratio_value_number_to_length (ratio_n_unique_to_length): done, test?
# root_mean_square: done, test?
# sample_entropy: done, test?
# spkt_welch_density ( Is this fourier related?): not yet

def ratio_n_unique_to_length(x: pl.Expr):
    return x.n_unique() / x.len()

def root_mean_square(x: pl.Expr) -> pl.Expr:
    '''
    Computes root mean square.

    Parameters
    ----------
    x : pl.Expr
        The input expression
    '''
    # Try (x.dot(x) / pl.count()).sqrt() and compare performance.
    # dot generally has pretty good performance
    return (x**2).mean().sqrt()

def range_count(x: pl.Expr, lower:float, upper:float) -> pl.Expr:
    '''
    Computes values of input expression that is between lower (inclusive) and upper (exclusive).

    Parameters
    ----------
    x : pl.Expr
        The input expression
    lower : float
        The lower bound, inclusive
    upper : float
        The upper bound, exclusive
    '''
    return x.is_between(lower_bound=lower, upper_bound=upper, closed="left").sum()

# I don't understand the doc for this.
def query_similarity_count(x:pl.Expr):
    pass

def ratio_beyond_r_sigma(x: pl.Expr, r: float) -> pl.Expr:
    '''
    Returns the ration of values in the series that is beyond r*std from mean on both sides.

    Parameters
    ----------
    x : pl.Expr
        Input expression
    r : float
        The scaling factor for std
    '''
    expr = (
        x.is_between(
            x.mean() - pl.lit(r) * x.std(),
            x.mean() + pl.lit(r) * x.std(),
            closed="both"
        ).is_not()
        .sum()
        / pl.count()
    )
    return expr

def _into_sequential_chunks(x:pl.Series, m:int) -> np.ndarray:
    '''
    '''

    cname = x.name
    n_rows = x.len() - m + 1
    df = x.to_frame().select(
        pl.col(cname)
        , *(pl.col(cname).shift(-i).suffix(str(i)) for i in range(1,m))
    ).slice(0, n_rows)
    return df.to_numpy()

def sample_entropy(x: pl.Series, r:float=0.2) -> float:
    # This is significantly faster than tsfresh

    threshold = r * x.std(ddof=0)
    m = 2
    mat = _into_sequential_chunks(x, m)
    tree = KDTree(mat)
    b = np.sum(tree.query_ball_point(mat, r = threshold, p = np.inf, workers=-1, return_length=True)) - mat.shape[0]
    mat = _into_sequential_chunks(x, m+1) #
    tree = KDTree(mat)
    a = np.sum(tree.query_ball_point(mat, r = threshold, p = np.inf, workers=-1, return_length=True)) - mat.shape[0]
    return np.log(b/a) # -ln(a/b) = ln(b/a)

def permutation_entropy(
    x: pl.Expr,
    tau:int=1,
    n_dims:int=3,
    base:float=math.e,
    normalize:bool=False
) -> pl.Expr:
    '''
    Computes permutation entropy.

    Paramters
    ---------
    tau : int
        The embedding time delay which controls the number of time periods between elements 
        of each of the new column vectors.
    n_dims : int, > 1
        The embedding dimension which controls the length of each of the new column vectors
    base : float
        The base for log in the entropy computation
    normalize : bool
        Whether to normalize in the entropy computation

    Reference
    ---------
        https://www.aptech.com/blog/permutation-entropy/
        https://tsfresh.readthedocs.io/en/latest/api/tsfresh.feature_extraction.html#tsfresh.feature_extraction.feature_calculators.permutation_entropy
    '''

    # CSE should take care of x.shift(-n_dims+1).is_not_null() ?
    # If input is eager, then in the divide statement we don't need need
    # a lazy expression to compute the remaining length.
    max_shift = -n_dims + 1
    out = (
        pl.concat_list(x, *(x.shift(-i) for i in range(1,n_dims))) # create list columns
        .take_every(tau) # take every tau
        .filter(x.shift(max_shift).is_not_null()) # This is a filter because length of df is unknown
        .list.eval(pl.element().rank(method="ordinal")) # for each inner list, do an argsort
        .value_counts() # groupby and count, but returns a struct
        .struct.field("counts") # extract the field named "counts"
        / x.shift(max_shift).is_not_null().sum() # get probabilities, alt:(pl.count() - max_shift)
    ).entropy(base=base, normalize=normalize).suffix("_permutation_entropy")

    return out
