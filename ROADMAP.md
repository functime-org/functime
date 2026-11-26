# `functime` Roadmap (`v1.x` → `v2.0`)

> **Tracking issue:** [`functime-org/functime#266`](https://github.com/functime-org/functime/issues/266) — detailed work items will be tracked via sub-issues and milestones.

## 1. End Goal (`v2.0`)

A lean, pure-Python, Polars-powered time-series toolkit working uniformly from a **single time series (`entity_col=None`) to 100,000+ series panels (`entity_col="id"`)**, paired with a **thin `functime.sklearn.*` wrapper layer** mirroring `scikit-learn` 1:1 and a **DI-first `functime.forecasting` module**.

Rather than maintaining hardcoded model wrappers (`lightgbm`, `catboost`, `xgboost`, `linear_model`, `flaml`, `lance`, and the already-removed `llm` module), users compose `functime.sklearn` components with `scikit-learn` pipelines and inject any external regressor (`LightGBM`, `XGBoost`, `CatBoost`, `Ridge`, etc.) into sklearn-compatible **`*Forecaster`** estimators under `functime.forecasting` (`RecursiveForecaster`, `DirectForecaster`, plus dedicated naive baselines such as `NaiveForecaster`, `SeasonalNaiveForecaster`, `MeanForecaster`, `DriftForecaster`) alongside functional **`backtest`** orchestration.

## 2. Architectural Pillars

1. **Two-Layer Separation & Shallow Domain Modules:** Pure functional Polars primitives hold 100% of domain logic with zero `scikit-learn` or Pandas imports (`functime._frame` / `functime._core` internally during `v1.x`, promoted to shallow top-level domain modules in `v2.0`: `preprocessing` — including seasonality — `feature_extraction`, `model_selection`, `metrics`, `diagnostics`, `conformal`, `plotting`). `functime.sklearn.*` strictly mirrors `scikit-learn` namespaces 1:1 (`preprocessing`, `impute`, `feature_extraction`, `model_selection`, `compose`, `metrics`), while `functime.forecasting` hosts sklearn-compatible `*Forecaster` estimators (`naive`, `recursive`, `direct`).
2. **Unified `Temporal` Contract & Two-Path Authoring:** Explicit `time_col: str` and `entity_col: str | list[str] | None = None` across all new APIs unify single series and multi-entity panels under one contract. Core authoring follows two paths sharing a single `@sklearn_transformer` wrapper on public `BaseTemporalTransformer`:
   - *Stateless (~85%)*: pure Polars expression functions decorated with `@temporal_transform(chronological=...)`, giving `.expr_fn(...)`, lazy `df.pipe(...)`, and 4-line `@sklearn_transformer(fn)` subclasses of `BaseTemporalTransformer`.
   - *Stateful*: plain pure-Polars core classes (`.fit(...) -> Self`, `.transform(...) -> pl.LazyFrame`, optional `.inverse_transform(...)`, zero `scikit-learn` imports) wrapped via `@sklearn_transformer(CoreClass)` on `BaseTemporalTransformer` (with direct `BaseTemporalTransformer` subclassing as an escape hatch).
3. **Lazy-First `TemporalFrame` Core & Eager `functime.sklearn` Boundary:** Core primitives accept any Narwhals-supported container (eager or lazy), parse once into a single always-lazy `TemporalFrame` (`pl.LazyFrame`), preserve chronological ordering via Polars-native `.over(entity_cols, order_by=time_col)` / `.over(pl.lit(0), order_by=time_col)` (zero eager sorts, preserving predicate pushdown), and always return `pl.LazyFrame`. `functime.sklearn.*` collects to `pl.DataFrame` at `transform` / `fit_transform` exit (avoiding multi-step `Pipeline` round-trips) and isolates `set_output(transform="pandas")` + Pandas index restoration exclusively inside `functime.sklearn`.
4. **Domain-Driven Design & Ubiquitous Language:** Establish and maintain `GLOSSARY.md` so unified `Temporal` terminology stays consistent across code, docs, and contributor PRs (replacing legacy `panel`-only naming).
5. **Minimal Core & Zero-Rust Local Setup (`#212`, `#238`):** Main `functime` package ships pure-Python (`py3-none-any`) in `v2.0`; custom Rust Polars kernels live in a separate optional package (`functime[rust]`), third-party plugins map directly (`functime[ols]` → `polars-ols`), and **`functime[fast]`** bundles all Rust/native accelerators.
6. **Lightweight Repo Data (`#239`):** Heavy M4/M5/Dunnhumby datasets live on Hugging Face ([`functime-org/datasets`](https://huggingface.co/datasets/functime-org/datasets)); only minimal private sample fixtures + synthetic generators stay in git.
7. **Code Organization, Dependencies & Strict Typing:** Top-down ("stepdown") module ordering (main public classes and entry points first, public methods before private hooks inside classes, reusable public functions next, private `_helpers` at the bottom), strict `pyrefly` checking, public-API-only NumPy docstrings, and explicit dependency floors (`polars>=1.25` in `v1.x` → `polars>=2.0` in `v2.0`, `narwhals>=2.22`, `typing_extensions>=4.16`, `scikit-learn>=1.6`), governed by [`AGENTS.md`](AGENTS.md).

## 3. Migration Path (`v1.x` → `v2.0`)

`functime` will ship the transition through **small, incremental `v1.x` releases** before cutting a single clean **`v2.0`** breaking release:

1. **`v1.x` — Foundations & Incremental New API (no removals):**
   - Maintain compatibility with `polars>=1.25` (both 1.x and 2.x), `narwhals>=2.22`, `typing_extensions>=4.16`, `scikit-learn>=1.6`, and Python `3.12–3.15` while keeping existing `v1` APIs working.
   - Establish and maintain `GLOSSARY.md` to enforce ubiquitous `Temporal` terminology across modules, documentation, and PR reviews.
   - Ship `functime._frame` / `functime._core` (`TemporalFrame`, `@temporal_transform`, pure-Polars stateful classes) and `functime.sklearn` (`BaseTemporalTransformer`, `@sklearn_transformer`), alongside new `PascalCase` `*Forecaster` classes inside `functime.forecasting` (`NaiveForecaster`, `SeasonalNaiveForecaster`, `MeanForecaster`, `DriftForecaster`, `RecursiveForecaster`, `DirectForecaster`) and non-colliding top-level modules (`model_selection`, `diagnostics`) progressively across small minor releases, alongside optional Rust decoupling (`functime[rust]`).
2. **Final `v1.x` — Formal Deprecations:**
   - Once the new interface (including `RecursiveForecaster`, `DirectForecaster`, naive forecasters, and `backtest`) covers the core workflows, mark legacy `v1` wrappers and modules (the `snake_case` hardcoded wrappers in `functime.forecasting`, `backtesting.py`, `@transformer`, legacy `preprocessing.py` and `seasonality/`) with `DeprecationWarning` pointing directly to their replacements and migration guidance.
3. **`v2.0` — Clean Pure-Python Break:**
   - Remove all deprecated `v1` wrappers/modules and the embedded Rust build (`py3-none-any` wheel).
   - Promote internal `functime._core` primitives to shallow top-level `functime.*` domain modules (`preprocessing` including seasonality, `feature_extraction`, `model_selection`, `metrics`, `diagnostics`, `conformal`, `plotting`) alongside `functime.sklearn.*` and the modernized `functime.forecasting`.
   - Bump minimum bounds cleanly (`polars>=2.0`, `narwhals>=2.22`, `typing_extensions>=4.16`, `scikit-learn>=1.6`) and eliminate the legacy `pyrefly` `basic` backlog (100% strict checking).
4. **`v2.x` — Additive Evolution:**
   - Expand purely additively on top of the stable `v2.0` API (e.g., ergonomic helpers such as optional `.backtest()` convenience methods if desired, additional diagnostics, and optional accelerators).
