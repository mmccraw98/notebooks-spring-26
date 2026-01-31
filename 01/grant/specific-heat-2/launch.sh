#!/usr/bin/env bash

set -euo pipefail

nv=20

for mu in 0.01 0.1 0.5 1.0; do
  for alpha in 1.0 1.2 1.5 2.0; do
    python jam.py --mu "$mu" --alpha "$alpha" --nv "$nv"
  done
done