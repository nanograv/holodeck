#!/bin/bash

for TARGET in hard_time gsmf_phi0
do 
    python ecg-notebooks/parameter_investigation/scripts/build_arrays_for_model.py $TARGET --favg -r 500 -v 21 -l 10 --bgl 1 
        
    for RED_GAMMA in -1.5 -3.0
    do
        for RED2WHITE in 0.5 1.0 2.0
        do
            python ecg-notebooks/parameter_investigation/scripts/build_arrays_for_model.py $TARGET --favg -r 500 -v 21 -l 10 --bgl 1 --red2white $RED2WHITE --red_gamma $RED_GAMMA
        done
    done
done