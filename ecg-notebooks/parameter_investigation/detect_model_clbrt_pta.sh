#!/bin/bash


for TARGET in hard_time gsmf_phi0 gsmf_mchar0_log10 mmb_mamp_log10 mmb_scatter_dex hard_gamma_inner
do 
    python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py $TARGET --detstats --debug -r 500 -s 100 -v 21 -l 1 
done
