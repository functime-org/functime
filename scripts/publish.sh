#!/usr/bin/env bash
set -e
set -x

for path in dist/*
do
    curl -F package=@${path} https://${GEMFURY_PUSH_TOKEN}@push.fury.io/functime/
done
