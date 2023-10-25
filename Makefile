.DEFAULT_GOAL := help

BASE ?= master
PY ?= python3

edit:
	maturin develop --release
	$(PY) -m pip install -e .

build:
	maturin develop --release
	$(PY) -m pip install .

build-test:
	maturin develop --release
	$(PY) -m pip install ".[test]"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	$(PY) -m pip uninstall functime -y

rebuild: clean build

.PHONY: help
help:  ## Display this help screen
  @echo -e '\033[1mAvailable commands:\033[0m'
  @grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
