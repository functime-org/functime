#!/usr/bin/env bash
set -e
set -x

mkdir coverage
source venv/bin/activate
coverage run -m pytest -v tests -m "not benchmark" --benchmark-disable ${@}
