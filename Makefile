.DEFAULT_GOAL := help

BASE ?= master
PY ?= python3

edit:
	$(PY) -m pip install -e .

build:
	$(PY) -m pip install .

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	$(PY) -m pip uninstall functime-client -y


.PHONY: help
help:  ## Display this help screen
  @echo -e '\033[1mAvailable commands:\033[0m'
  @grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
