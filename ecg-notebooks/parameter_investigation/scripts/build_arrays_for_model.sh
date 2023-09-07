#!/bin/bash

for TARGET in gsmf_phi0 gsmf_mchar0_log10 mmb_mamp_log10 mmb_scatter_dex
do 
    python ecg-notebooks/parameter_investigation/scripts/build_arrays_for_model.py $TARGET --anis_freq -r 500 -v 3 -l 1000 

done
