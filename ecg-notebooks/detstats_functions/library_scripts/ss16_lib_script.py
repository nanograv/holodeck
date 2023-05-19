#!/usr/bin/env python3

import holodeck.detstats as ds
import h5py

LIB_PATH = '/Users/emigardiner/GWs/holodeck/output/2023-05-12-mbp-ss16_n10_r10_f70_d12.5_l10_p0'
hdf_name = LIB_PATH+'/ss_lib.hdf5'
print('Hdf file:', hdf_name)

# settings to change
DS_NUM = '03'
NPSRS = 67
SIGMA = 4e-7

NSKIES = 25
THRESH = 0.5
PLOT = True
DEBUG = True

output_dir = (LIB_PATH+'/detstats/ds%s_psrs%d_sigma%.0e'
              % (DS_NUM, NPSRS, SIGMA))
print('Output dir:', output_dir)

vals = ds.detect_lib(hdf_name, output_dir, NPSRS, SIGMA, 
                     nskies=NSKIES, thresh=THRESH, plot=PLOT, debug=DEBUG)