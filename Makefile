.DEFAULT_GOAL := help

BASE ?= master

build:
	pip install -e .

clean:
	rm -rf *.egg-info


.PHONY: help
help:  ## Display this help screen
  @echo -e '\033[1mAvailable commands:\033[0m'
  @grep -E '^[a-z.A-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' | sort
