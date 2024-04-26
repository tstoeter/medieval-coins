#!/usr/bin/bash

set -eo pipefail

for acc in $(seq 0 9)
do
	for qnt in $(seq 0 30)
	do
		python3 analyse_detections.py $acc $qnt
	done
done

