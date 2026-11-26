# Agent & Contributor Guidelines

See [`ROADMAP.md`](ROADMAP.md) for architecture and the `v1.x` → `v2.0` migration path, and [`GLOSSARY.md`](GLOSSARY.md) for the project's ubiquitous domain language.

## Architecture Principles

- **Core first:** Put all domain logic in pure-Polars primitives (`functime._frame` / `functime._core` during `v1.x`, top-level `functime.*` in `v2.0`) with zero `scikit-learn` or Pandas imports. Keep `functime.sklearn.*` strictly as thin 1:1 `scikit-learn` wrappers (via `BaseTemporalTransformer` and `@sklearn_transformer`) and `functime.forecasting.*` as sklearn-compatible `*Forecaster` estimators, all passing `parametrize_with_checks` (with explicit `expected_failed_checks` where 2D-NumPy-only checks do not apply).
- **Univariate + panel:** New APIs take explicit `time_col: str` and `entity_col: str | list[str] | None = None`. Never assume positional columns.
- **Lazy Polars core, eager `sklearn` boundary:** Convert Narwhals-supported containers (eager or lazy) at entry into a single always-lazy `TemporalFrame` (`pl.LazyFrame`), execute natively in Polars (preserving chronological order via `.over(..., order_by=...)`), and always return `pl.LazyFrame` from core primitives. Collect to `pl.DataFrame` at `functime.sklearn` `transform` / `fit_transform` exit and isolate `set_output(transform="pandas")` + Pandas index restoration inside `functime.sklearn`.

## Code & Documentation Hygiene

- **Top-down module ordering:** Order modules top-down ("stepdown"): main public classes and entry points at the top (public methods before private hooks inside classes), reusable public functions below them, and private `_helpers` at the bottom.
- **Public docstrings only:** Write concise NumPy-style docstrings on public classes/functions. Do **not** write docstrings on private (`_helper`) functions or methods.
- **Comments explain *why*, never *what*:** Comment only non-obvious invariants, edge cases, or Polars/`scikit-learn` workarounds. Remove redundant narrating comments.
- **Minimal dependencies:** Never add unapproved core dependencies. Keep Rust/external Polars plugins optional.

## Issues, PRs & AI Disclosure

- **Issue first:** Open an issue before submitting a PR to discuss features or design changes — and include a minimal reproducible example (**reprex**) for bug reports. Unsolicited PRs without a linked issue may be closed.
- **Accountability & AI disclosure:** Using AI coding agents / LLMs is welcome, but it does not reduce individual contributor accountability. Contributors are responsible for understanding and validating everything they submit, and **LLM/agent usage must be disclosed** in the PR.
- **Stacked PRs:** Use `gh stack` to manage stacked PRs. Never embed manual stack tables or cross-PR lists in PR descriptions.
- **PR descriptions:**
  - Start with **Why**: the current problem and the goal of the PR.
  - Include a section showing the **Change** (e.g., before/after API usage, behavior diff, or benchmark results).
  - Do **not** write file-by-file change lists — reviewers read the diff for that.
  - **Never** include a "Verification / ran tests locally" section — running tests locally is a baseline expectation, and lint/type/test checks are enforced deterministically by pre-commit and CI.
