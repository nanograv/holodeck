#!/usr/bin/env python3

import holodeck.detstats as ds
import h5py

LIB_PATH = '/Users/emigardiner/GWs/holodeck/output/2023-05-18-awg-sam01_uniform07B_n1000_r1000'
hdf_name = LIB_PATH+'/sam_lib.hdf5'
print('Hdf file:', hdf_name)

# settings to change, manually calibrated
DS_NUM = '01'
NPSRS = 60
SIGMA = 1.24e-8 

NSKIES = 25
THRESH = 0.5
PLOT = True
DEBUG = True

def main():
    output_dir = (LIB_PATH+'/detstats/ds%s_psrs%d_sigma%.0e'
                % (DS_NUM, NPSRS, SIGMA))
    print('Output dir:', output_dir)

    vals = ds.detect_lib(hdf_name, output_dir, NPSRS, SIGMA, 
                        nskies=NSKIES, thresh=THRESH, plot=PLOT, debug=DEBUG)
    
if __name__ == "__main__":
    main()