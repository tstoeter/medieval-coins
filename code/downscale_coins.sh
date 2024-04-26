#!/usr/bin/bash

set -eo pipefail

find $base/data/ -name "*_normal.tif" -print0 | xargs -0 -n 1 -P $(nproc) python3 downscale_coin.py 10

