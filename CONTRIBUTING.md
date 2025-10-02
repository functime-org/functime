# Contributing to functime

Thank you for considering contributing to functime! We value every contribution, whether it's a bug report, a feature request, or a code contribution.

## Development Setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- [just](https://github.com/casey/just) - Command runner
- [Rust toolchain](https://rustup.rs/) - For building the Rust extension module
- Python 3.9 or higher

### Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/functime-org/functime.git
   cd functime
   ```

2. **Install uv** (if not already installed)
   Follow the [official uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

3. **Install just** (if not already installed)
   See the [just installation instructions](https://just.systems/man/en/chapter_4.html) for your platform.

4. **Sync dependencies**
   ```bash
   uv sync --all-extras --dev --locked
   ```

5. **Build the project**
   ```bash
   just build
   ```

## Development Workflow

### Common Commands

Run `just` or `just --list` to see all available commands:

- `just build` - Build the extension in debug mode for fast local iterations
- `just build-release` - Produce the optimised build that matches CI artefacts
- `just test` - Run the test suite
- `just format` - Format code and sort imports
- `just lint` - Lint the code
- `just clean` - Clean build artefacts and caches
- `just rebuild` - Clean and perform a release build
- `just publish-dry-run` - Exercise the publishing pipeline without uploading

### Running Tests

```bash
# Run all tests (excluding benchmarks and slow tests)
just test

# Run specific tests
just test tests/test_preprocessing.py

# Run with additional pytest arguments
just test -- -k "test_specific_function" -v
```

### Code Quality

Before submitting a PR, ensure your code passes formatting and linting:

```bash
# Format code
just format

# Check linting
just lint

# Fix linting issues automatically
just lint --fix
```

### Building the Documentation

```bash
# Install doc dependencies
uv sync --group doc --locked

# Serve docs locally
uv run mkdocs serve
```

### Running Benchmarks

```bash
just benchmark
```

## Making Changes

1. Create a new branch for your feature or bugfix
2. Make your changes
3. Add tests for new functionality
4. Run `just format` and `just lint` to ensure code quality
5. Run `just test` to ensure all tests pass
6. Commit your changes with clear commit messages
7. Push to your fork and submit a pull request

## Code Style

functime uses:
- **Ruff** for linting and formatting
- Imports are automatically sorted
- Line length: 88 characters (Black-compatible)

All formatting is handled by the `just format` command.

## Project Structure

```
functime/
├── functime/          # Python source code
├── src/               # Rust source code
├── tests/             # Test files
├── docs/              # Documentation
├── pyproject.toml     # Project configuration
├── justfile           # Task automation
└── Cargo.toml         # Rust project configuration
```

## Testing Philosophy

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names
- Group related tests using markers

## Dependency Management

functime uses uv for dependency management:

- **Add a dependency**: `uv add <package>` followed by `uv lock` and commit the resulting `uv.lock`
- **Add a dev dependency**: `uv add --group dev <package>` then `uv lock`
- **Sync dependencies**: `uv sync --dev --locked`
- **Update dependencies**: `uv lock --upgrade` and commit the updated lockfile

## Need Help?

- Check the [documentation](https://docs.functime.ai)
- Join our [Discord](https://discord.com/invite/JKMrZKjEwN)
- Open an issue on GitHub

## Code of Conduct

Please note that this project follows our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## License

By contributing to functime, you agree that your contributions will be licensed under the Apache-2.0 License.
