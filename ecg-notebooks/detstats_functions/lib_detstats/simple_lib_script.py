#!/usr/bin/env python3

import holodeck.detstats as ds
import h5py

HDF_NAME = '/Users/emigardiner/GWs/holodeck/output/2023-05-09-mbp-ss14_n40_r10_f20_d17.5_l5_p0/ss_lib.hdf5'
print('Hdf file:', HDF_NAME)

# settings to change
DS_NUM = 3
NPSRS = 40
SIGMA = 1e-7

NSKIES = 25
THRESH = 0.5
PLOT = True
DEBUG = True

OUTPUT_DIR = ('/Users/emigardiner/GWs/holodeck/output/2023-05-09-mbp-ss14_n40_r10_f20_d17.5_l5_p0/detstats/ds%d_psrs%d_sigma%.0e'
              % (DS_NUM, NPSRS, SIGMA))
print('Output dir:', OUTPUT_DIR)

vals = ds.detect_lib(HDF_NAME, OUTPUT_DIR, NPSRS, SIGMA, 
                     nskies=NSKIES, thresh=THRESH, plot=PLOT, debug=DEBUG)