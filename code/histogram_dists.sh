#!/usr/bin/bash

set -eo pipefail

find $base/output/correct_bending/ -name "*_corr_bending.npy" -print0 | xargs -0 -n 1 -P $(nproc) python3 histogram_dist.py

