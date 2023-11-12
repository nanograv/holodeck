# import
import os
import glob
from enterprise.pulsar import Pulsar
import pickle

N_REAL = 2     # number of realizations to produce
N_PSRS = 40      # use fewer pulsars for debugging
RR = 0

if RR >= N_REAL:
    err = f"realization index {RR=} too high for {N_REAL=}"
    raise ValueError(err)

save_dir = "/Users/emigardiner/GWs/holodeck/output/holodeck_extension_15yr_stuff/2023_11_07_sam_datasets_15yr_based_r002_v01"
# save_dir = f"/Users/emigardiner/GWs/holodeck/output/holodeck_extension_15yr_stuff/Test_sam_datasets_15yr_based_r{N_REAL:03d}_v01/"
real_dir = save_dir+"/real{0:03d}/".format(RR)
if os.path.exists(real_dir) is False:
    err = f"{real_dir=} does not exist"
    raise ValueError(err)

#get list of par files
timfiles = sorted(glob.glob(real_dir + '/*.tim'))
parfiles = sorted(glob.glob(save_dir + '/*.par'))
if N_PSRS is not None:
    timfiles = timfiles[:N_PSRS]
    parfiles = parfiles[:N_PSRS]
assert len(timfiles)==len(parfiles), "mismatch between number of par and tim files!"
print(f"{timfiles=}\n{parfiles=}")

# load each psr
psrs = []
for par, tim in zip(parfiles, timfiles):
    print(f"{tim=}")
    psr = Pulsar(par, tim)
    psrs.append(psr)

# pickle and save
pkl_file = real_dir + f'data_15yr_fake_r{RR:03d}_of_{N_REAL:03d}_p{N_PSRS}_v01.pkl'
print(pkl_file)
with open(pkl_file, 'wb') as file: 
    # A new file will be created 
    pickle.dump(psrs, file) 