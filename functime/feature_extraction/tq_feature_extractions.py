import polars as pl
from scipy.spatial import KDTree
import numpy as np
import math

# permutation_entropy: done, tested against tsfresh
# query_similarity_count: done, but not sure if this is correct...
# range_count: done, tested
# ratio_beyond_r_sigma: done, don't think I need to test this
# ratio_value_number_to_length (ratio_n_unique_to_length): done, don't think I need to test this
# root_mean_square: done, don't think I need to test this
# sample_entropy: done, tested against tsfresh
# spkt_welch_density ( Is this fourier related?): not yet

def spkt_welch_density():
    pass

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
    # Using dot is usually a few micro-seconds faster
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
    if upper < lower:
        raise ValueError("Upper must be greater than lower.")
    return x.is_between(lower_bound=lower, upper_bound=upper, closed="left").sum()

# I have no idea how to test this. This is purely based on my understanding of what I read
# from tsfresh. I do not know how to use tsfresh's query_similarity_count either.
def query_similarity_count(x:pl.Expr, subseq:list[float], threshold:float, z_normalize:bool=False):
    '''
    Count how many subsequences of x are similar to the given subseq. Euclidean distance is used.

    Parameters
    ----------
    x : pl.Expr
        The input expression
    subseq : list[float] | Series
        A list of floats or a Polars series to compare with
    threshold : float
        If distance is within threshold, it is considered a similar match.
    z_normalize: bool
        Whether to z_normalize before comparing.
    '''
    # If we want to allow other distances metrics, one way 
    # is to change pl.element().pow(2).sum().sqrt() step

    s = pl.Series(subseq, dtype=pl.Float64)
    if z_normalize:
        s = (s - s.mean())/s.std()
        mean = pl.concat_list(x, *(x.shift(-i) for i in range(1, len(s)))).list.eval(pl.element().mean()).list.first()
        std = pl.concat_list(x, *(x.shift(-i) for i in range(1, len(s)))).list.eval(pl.element().std()).list.first()
        expr = (
            pl.concat_list((x-mean)/std - pl.lit(s[0]), *(((x.shift(-i)-mean)/std - pl.lit(s[i])) 
                                                          for i in range(1, len(s))))
            .list.eval(pl.element().pow(2).sum().sqrt())
            .list.first().lt(threshold).sum()
        )
    else:
        expr = (
            pl.concat_list(x - s[0], *(x.shift(-i) - s[i] for i in range(1, len(s))))
            .list.eval(pl.element().pow(2).sum().sqrt())
            .list.first().lt(threshold).sum()
        )

    return expr

def ratio_beyond_r_sigma(x: pl.Expr, r: float) -> pl.Expr:
    '''
    Returns the ratio of values in the series that is beyond r*std from mean on both sides.

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
        ).is_not() # check for deprecation
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

    # This is kind of slow rn when tau > 1
    max_shift = -n_dims + 1
    out = (
        pl.concat_list(x, *(x.shift(-i) for i in range(1,n_dims))) # create list columns
        .take_every(tau) # take every tau
        .filter(x.shift(max_shift).take_every(tau).is_not_null()) # This is a filter because length of df is unknown, might slow down perf
        .list.eval(pl.element().rank(method="ordinal")) # for each inner list, do an argsort
        .value_counts() # groupby and count, but returns a struct
        .struct.field("counts") # extract the field named "counts"
        / x.shift(max_shift).take_every(tau).is_not_null().sum() # get probabilities
    ).entropy(base=base, normalize=normalize).suffix("_permutation_entropy")

    return out
