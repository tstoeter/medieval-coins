#!/usr/bin/bash

# from https://stackoverflow.com/questions/4774054/reliable-way-for-a-bash-script-to-get-the-full-path-to-itself
export base="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"


### retrieve dataset ###

# download and unpack datasets
cd $base/data
time bash download_unpack.sh


### preprocessing ###

# conversion and downscaling coins
cd $base/code
time bash downscale_coins.sh

cd $base/code
time bash correct_bendings.sh


### global similarity ###
cd $base/code
time bash histogram_dists.sh


# parameter study for coin detection
cd $base/code
time bash detect_coins.sh

# parameter analysis with detected vs. labelled coins
cd $base/code
echo "#accum_idx quantile_idx quantile_value accum_mean hit_mean hit2_mean dist_center dist_inner_mean dist_outer_mean" > detections_parameters.txt
time bash analyse_detections.sh >> detections_parameters.txt

time python3 plot_analysed_detections.py
time python3 plot_labels_detections.py
mv detections_parameters* $base/results/
mv detections_*.pdf $base/results/

# compare coins wrt various similarity measures
cd $base/code
time bash compare_coins.sh


### analysis ###

cd $base/code
time python3 analyse_histogram_dist.py
time python3 analyse_globalsim.py $base/output/global_sim_rot/
mv *.pdf $base/results/

