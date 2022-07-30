"""Observational Data

This module provides interfaces to access observational data and constraints, primarily for use
in calibration and diagnostics of models.

To-Do
-----

"""
import os

import numpy as np

import holodeck as holo

FNAME_MM2013 = "mcconnell+ma-2013_1211.2816.txt"


def load_mcconnell_ma_2013():
    fname = os.path.join(holo._PATH_DATA, FNAME_MM2013)
    header = []
    data_raw = []
    hcnt = 0
    cnt = 0
    for line in open(fname, 'r').readlines():
        line = line.strip()
        if (len(line) == 0) or (cnt < 2):
            cnt += 1
            continue

        if line.startswith('Col'):
            line = line.split(':')[-1].strip()
            header.append(line)
            cnt += 1
            hcnt += 1
            continue

        line = [ll.strip() for ll in line.split()]
        for ii in range(len(line)):
            try:
                line[ii] = float(line[ii])
            except ValueError:
                pass

        data_raw.append(line)
        cnt += 1

    data = dict()
    data['name'] = np.array([dr[0] for dr in data_raw])
    data['dist'] = np.array([dr[1] for dr in data_raw])
    data['mass'] = np.array([[dr[3], dr[2], dr[4]] for dr in data_raw])
    data['sigma'] = np.array([[dr[7], dr[6], dr[8]] for dr in data_raw])
    data['lumv'] = np.array([[dr[9], dr[10]] for dr in data_raw])
    data['mbulge'] = np.array([dr[13] for dr in data_raw])
    data['rinf'] = np.array([dr[14] for dr in data_raw])
    data['reffv'] = np.array([dr[17] for dr in data_raw])

    return data
