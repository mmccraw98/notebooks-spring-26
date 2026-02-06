#!/usr/bin/env bash

set -euo pipefail

nv=40

for asperity_radius in 0.1 0.15 0.2 0.3 0.4; do
  python jam.py --asperity_radius "$asperity_radius" --nv "$nv"
done