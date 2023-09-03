#!/bin/bash


python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py hard_time --detstats --debug -r 500 -s 100 -v 21 --cv 10 --gsc-clbrt --divide

python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py gsmf_phi0 --detstats --debug -r 500 -s 100 -v 21 --cv 10 --gsc-clbrt --divide

python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py gsmf_mchar0_log10 --detstats --debug -r 500 -s 100 -v 21 --cv 10 --gsc-clbrt --divide

python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py mmb_mamp_log10 --detstats --debug -r 500 -s 100 -v 21 --cv 10 --gsc-clbrt --divide

python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py mmb_scatter_dex --detstats --debug -r 500 -s 100 -v 21 --cv 10 --gsc-clbrt --divide

python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py hard_gamma_inner --detstats --debug -r 500 -s 100 -v 21 --cv 10 --gsc-clbrt --divide
