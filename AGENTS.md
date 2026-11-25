# Agent & Contributor Guidelines

See [`ROADMAP.md`](ROADMAP.md) for architecture and the `v1.x` → `v2.0` migration path.

## Architecture Principles

- **Core first:** Put all domain logic in pure functional Polars primitives (`functime._core.*` during `v1.x`, top-level `functime.*` in `v2.0`). Keep `functime.sklearn.*` strictly as thin 1:1 `scikit-learn` wrappers and `functime.forecasting.*` as sklearn-compatible `*Forecaster` estimators, all passing `parametrize_with_checks` (with explicit `expected_failed_checks` where 2D-NumPy-only checks do not apply).
- **Univariate + panel:** New APIs take explicit `time_col: str` and `entity_col: str | list[str] | None = None`. Never assume positional columns.
- **Narwhals at boundaries, Polars inside:** Convert Narwhals inputs to Polars at entry, execute natively in Polars, return `pl.DataFrame` by default, and honor `set_output`.

## Code & Documentation Hygiene

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
