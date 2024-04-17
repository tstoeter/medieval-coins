#!/usr/bin/bash

find $base/output/coins_10ppmm/ -name "*.npy" -print0 | xargs -0 -n 1 -P $(nproc) python3 correct_bending.py

