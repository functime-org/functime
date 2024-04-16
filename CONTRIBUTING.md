Thanks for taking the time to contribute! We appreciate all contributions, from reporting bugs to implementing new features. If it's unclear on how to proceed after reading this guide, you can ask on [Discord](https://discord.gg/dNfGMUyPa8).

This guide covers the following:

1. [Reporting an issue or a feature request](#opening-an-issue).
2. [Contributing to the code base](#contributing-to-the-codebase).
    * [Picking an issue](#picking-an-issue)
    * [Set up your local environment](#set-up-your-local-environment)
    * [While working on your issue](#while-working-on-your-issue)
    * [Pull requests](#pull-requests)
3. [Contributing to the documentation](#contributing-to-the-documentation).
4. [Credits](#credits)

# Opening an issue

You can report any issue by opening a [new issue](https://github.com/functime-org/functime/issues/new/choose).

**Bug reports** should include:

1. Your **OS, the Python version and `functime` version** you are using.
2. A **minimal reproducible example (MRE)**, i.e. the code and some (fake) data that can be used to reproduce the error you encounter. It might take a bit more time on your side, but it greatly helps maintainers to solve your issue quickly.

**Feature requests** should also start from a dedicated issue, even if you plan to contribute to the feature yourself. In this way, maintainers can help you plan the design of the new feature and ease the development.

# Contributing to the codebase

Contributions should always start from an issue: even if you wish to contribute to `functime`'s  features, it is best to open a new issue so that the maintainers can help you through the design process.

## Picking an issue

Pick an issue by going through the [issue tracker](https://github.com/functime-org/functime/issues) and finding an issue you would like to work on. To work on an issue, please leave a new message below the discussion to show your interest. We use the [`help wanted`](https://github.com/functime-org/functime/labels/help%20wanted) label to indicate issues that are high on our wishlist. However, if you are a first time contributor, you might want to look for issues labeled [`good first issue`](https://github.com/functime-org/functime/labels/good%20first%20issue).

## Set up your local environment

This might be slightly complex, because `functime` uses some Rust plugins to accelerate some features. In other words, you need to make sure you have installed both [Python](https://www.python.org/) and [Rust](https://www.rust-lang.org/) on your machine (see below).

>[!NOTE] This might raise unexpected issue if you use Windows. For that case, it would be best if you develop using [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to install Linux on Windows.

1. **Fork the repository**. you can follow [this guide](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo). For example, using the GitHub CLI, you would just need to do this:

```bash
gh repo fork functime-org/functime
```

The CLI will prompt you to clone the fork locally.

2. **Clone the repository locally**.

```bash
# via gh CLI
gh repo clone functime-org/functime

# via https
git clone https://github.com/<your-username>/functime

# via ssh (safer)
git clone git@github.com:functime-org/functime
```

3. **Install Rust**. This is easily done with [`rustup`](https://rustup.rs/). Use the latest stable version.

3. **Install Python**. Since `functime` depends on some packages from Python's scientific ecosystem, we respect **numpy's minimum supported version** (see [here](https://numpy.org/neps/nep-0029-deprecation_policy.html#support-table)). Though you can download Python from the [official page](https://www.python.org/downloads/), **we recommend you use [`rye`](https://rye-up.com/) to manage your Python versions and install the project dependencies**. This will make the next installation step easier. You can also use [`pdm`](https://pdm-project.org/en/latest/) or [`hatch`](https://hatch.pypa.io/1.9/). `poetry` will not work, as it does not comply with with PEP517 and PEP518.

4. **Install the project's dependencies**. If you use `rye`, run the following:

```bash
# with rye
rye sync --features=dev
```

5. Install pre-commit hooks:

```bash
rye run pre-commit install --install-hooks
```

## While working on your issue

Create a new git branch from the `main` branch in your local repository, and start coding!

The Rust code is located in the `src` directory, while the Python codebase is located under `functime`. To run the tests, use the following:

```bash
rye test
```

`pre-commit` checks will run before any commit. To format the code, use the following:

```bash
rye fmt
rye lint
```

Note that your work cannot be merged if these checks fail!

Two other things to keep in mind:

* Add test to your code. If you haven't written tests before, the dev team will be glad to help you out. We will link some useful resources here too.
* If you change the public API, update the documentation.

## Pull requests

When you have resolved your issue, [open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) in the repository. Please adhere to the following guidelines:

* **Start your pull request title with a [conventional commit tag](https://www.conventionalcommits.org/en/v1.0.0/)**. This helps us add your contribution to the right section of the changelog. We use the [Angular](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#type) convention.
* Use a descriptive title starting with an uppercase letter. This text will end up in the changelog.
* In the pull request description, [link](https://docs.github.com/en/issues/tracking-your-work-with-issues/linking-a-pull-request-to-an-issue) to the issue you were working on.
* Add any relevant information to the description that you think may help the maintainers review your code.
* Make sure your branch is [rebased](https://docs.github.com/en/get-started/using-git/about-git-rebase) against the latest version of the main branch.
* Make sure all GitHub Actions checks pass.
* After you have opened your pull request, a maintainer will review it and possibly leave some comments. Once all issues are resolved, the maintainer will merge your pull request, and your work will be part of the next functime release!

Keep in mind that your work does not have to be perfect right away! If you are stuck or unsure about your solution, feel free to open a draft pull request and ask for help.

# Contributing to the documentation

*In progress...*

# Credits

This guide is inspired from [Polars User guide](https://docs.pola.rs/development/contributing/).
