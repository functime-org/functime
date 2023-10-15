from .automl import (
    auto_elastic_net,
    auto_knn,
    auto_lasso,
    auto_linear_model,
    auto_ridge,
)
from .censored import censored_model, zero_inflated_model
from .elite import elite
from .knn import knn
from .linear import (
    elastic_net,
    elastic_net_cv,
    lasso,
    lasso_cv,
    linear_model,
    ridge,
    ridge_cv,
)
from .naive import naive
from .snaive import snaive

try:
    from .lance import ann
except ImportError:
    msg = "Missing ann extras: `pip install functime[ann]`"
    ann = ImportError(msg)

try:
    from .automl import auto_lightgbm
    from .lightgbm import flaml_lightgbm, lightgbm
except ImportError:
    msg = "Missing lightgbm extras: `pip install functime[lgb]`"
    auto_lightgbm = ImportError(msg)
    flaml_lightgbm = ImportError(msg)
    lightgbm = ImportError(msg)

try:
    from .catboost import catboost
except ImportError:
    catboost = ImportError("Missing catboost extras: `pip install functime[cat]`")

try:
    from .xgboost import xgboost
except ImportError:
    xgboost = ImportError("Missing xgboost extras: `pip install functime[xgb]`")


__all__ = [
    "ann",
    "auto_elastic_net",
    "auto_knn",
    "auto_lasso",
    "auto_lightgbm",
    "auto_linear_model",
    "auto_ridge",
    "catboost",
    "censored_model",
    "elastic_net_cv",
    "elastic_net",
    "elite",
    "flaml_lightgbm",
    "knn",
    "lasso_cv",
    "lasso",
    "lightgbm",
    "linear_model",
    "naive",
    "ridge_cv",
    "ridge",
    "snaive",
    "xgboost",
    "zero_inflated_model",
]
