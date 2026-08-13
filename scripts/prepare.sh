#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
python -m pgllmkt.cli doctor
python -m pgllmkt.cli prepare "$@"
