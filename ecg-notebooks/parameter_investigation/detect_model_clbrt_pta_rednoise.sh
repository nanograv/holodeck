#!/bin/bash
    

for TARGET in hard_time gsmf_phi0 
do 
    for RED_GAMMA in -1.5 -3.0
    do
        python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py $TARGET --construct --debug -r 500 -v 21 -l 10 --bgl 1 --red2white 1.0 --red_gamma $RED_GAMMA
    done
done

for TARGET in gsmf_mchar0_log10 mmb_mamp_log10 mmb_scatter_dex hard_gamma_inner
do 
    for RED_GAMMA in -1.5 -3.0
    do
        for RED2WHITE in 0.5 1.0 2.0
        do
            python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py $TARGET --construct --debug -r 500 -v 21 -l 10 --bgl 1 --red2white $RED2WHITE --red_gamma $RED_GAMMA
        done
    done
done

# red noise for gw only
# gw only
# for TARGET in gsmf_phi0 gsmf_mchar0_log10 mmb_mamp_log10 mmb_scatter_dex 
# do 
#     python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py $TARGET --detstats --debug -r 500 -s 100 -v 21 -l 10 --bgl 1 --red2white 0.5 --red_gamma -1.5 --gw_only
#     python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py $TARGET --detstats --debug -r 500 -s 100 -v 21 -l 10 --bgl 1 --red2white 0.5 --red_gamma -3.0 --gw_only
#     python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py $TARGET --detstats --debug -r 500 -s 100 -v 21 -l 10 --bgl 1 --red2white 2.0 --red_gamma -1.5 --gw_only
#     python ecg-notebooks/parameter_investigation/detect_model_clbrt_pta.py $TARGET --detstats --debug -r 500 -s 100 -v 21 -l 10 --bgl 1 --red2white 2.0 --red_gamma -3.0 --gw_only

# done