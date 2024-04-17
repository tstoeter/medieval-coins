#!/usr/bin/bash

find $base/output/correct_bending/ -name "*corr_bending.npy" -print0 | xargs -0 -n 1 -P $(nproc) python3 compare_coin_rot.py

