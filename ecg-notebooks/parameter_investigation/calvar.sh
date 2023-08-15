#!/bin/bash

for target in hard_time gsmf_phi0 gsmf_mchar0_log10 mmb_mamp_log10 mmb_scatter_dex hard_gamma_inner

do
    echo $target
    python ecg-notebooks/parameter_investigation/detect_model_varclbrt.py $target --detstats --debug -r 500 -s 100 -v 21 --cv 0

done