#!/usr/bin/env python3

import holodeck.detstats as ds
import h5py

LIB_PATH = '/Users/emigardiner/GWs/holodeck/output/2023-05-09-mbp-ss15_n100_r30_f100_d15_l5_p0'
hdf_name = LIB_PATH+'/ss_lib.hdf5'
print('Hdf file:', hdf_name)

# settings to change
DS_NUM = '2B'
NPSRS = 40
SIGMA = 5e-7

NSKIES = 25
THRESH = 0.5
PLOT = True
DEBUG = True

output_dir = (LIB_PATH+'/detstats/ds%s_psrs%d_sigma%.0e'
              % (DS_NUM, NPSRS, SIGMA))
print('Output dir:', output_dir)

vals = ds.detect_lib(hdf_name, output_dir, NPSRS, SIGMA, 
                     nskies=NSKIES, thresh=THRESH, plot=PLOT, debug=DEBUG)