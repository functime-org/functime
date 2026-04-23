set quiet := true

alias fmt := format
alias check := lint

[group("Build")]
[doc("Build the extension in debug mode (fast rebuilds)")]
build:
    uv run --with "maturin>=1.5" -- maturin develop

[group("Build")]
[doc("Build the extension in release mode (optimised)")]
build-release:
    uv run --with "maturin>=1.5" -- maturin develop --release

[group("Build")]
[doc("Clean then perform a release build")]
rebuild: clean build-release

[group("Build")]
[doc("Remove build artefacts")]
clean:
    rm -rf build/ dist/ *.egg-info target/

[group("Dev")]
[doc("Install all dev dependencies using the lockfile")]
sync:
    uv sync --all-extras --dev --locked

[group("Dev")]
[doc("Format code and sort imports")]
format *path=".":
    uv run --with ruff -- ruff format {{ path }}
    uv run --with ruff -- ruff check --fix --unsafe-fixes --select I {{ path }}

[group("Dev")]
[doc("Lint code without autofix")]
lint *args=".":
    uv run --with ruff -- ruff check {{ args }}

[group("Test")]
[doc("Run the default test suite (benchmarks excluded)")]
test *args="":
    uv run pytest tests -vv --show-capture=no --tb=line --benchmark-disable -k "not test_benchmarks" {{ args }}

[group("Test")]
[doc("Run benchmark-only pytest selection")]
benchmark *args="":
    uv run pytest -k "test_benchmark" -vv \
        --benchmark-name=short \
        --benchmark-group-by=param \
        --benchmark-min-rounds=3 \
        --benchmark-columns=min,max,mean,median,stddev,rounds,iterations {{ args }}

[group("Release")]
[doc("Dry-run publish to PyPI using the staged artefacts")]
publish-dry-run:
    UV_PUBLISH_TOKEN=${UV_PUBLISH_TOKEN:-""} uv publish --dry-run

[group("Test")]
[doc("Run quickstart scripts against the built wheel")]
quickstart:
    uv run python docs/code/quickstart.py
    uv run python docs/code/feature_engineering.py
