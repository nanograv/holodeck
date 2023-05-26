#!/usr/bin/env python3

import holodeck as holo
import holodeck.detstats as ds
from holodeck.constants import YR
import numpy as np

LIB_PATH = '/Users/emigardiner/GWs/holodeck/output/awg_output/2023-05-18-awg-sam01_uniform07B_n1000_r1000'
hdf_name = LIB_PATH+'/sam_lib.hdf5'
print('Hdf file:', hdf_name)

# settings to change, manually calibrated
DS_NUM = '01'
NPSRS = 67
SIGMA = 2.85e-6 

NSKIES = 25
THRESH = 0.5
PLOT = True
DEBUG = True

FIND_DUR = True

def main():
    output_dir = (LIB_PATH+'/detstats/ds%s_psrs%d_sigma%.0e'
                % (DS_NUM, NPSRS, SIGMA))
    print('Output dir:', output_dir)

    if(FIND_DUR):
        nfreqs = 40# hardcoded 
        dur = holo.librarian.DEF_PTA_DUR * YR # hardcoded
        hifr = nfreqs/dur
        cad = 1.0 / (2 * hifr)
        fobs_edges = holo.utils.nyquist_freqs_edges(dur, cad)
        dfobs = np.diff(fobs_edges)

    vals = ds.detect_lib(hdf_name, output_dir, NPSRS, SIGMA, 
                        nskies=NSKIES, thresh=THRESH, plot=PLOT, debug=DEBUG,
                        dur=dur, cad=cad, dfobs=dfobs)
    
    print('Detstats complete!')
    
if __name__ == "__main__":
    main()