"""
"""

import h5py
import numpy as np
import tqdm

import holodeck as holo


def sam_lib_combine(path_output, log, debug=False):
    log.info(f"{path_output=}")

    regex = "lib_sams__p*.npz"
    files = sorted(path_output.glob(regex))
    num_files = len(files)
    log.info(f"{path_output=}\n\texists={path_output.exists()}, found {num_files} files")

    all_exist = True
    log.info("Checking files")
    for ii in tqdm.tqdm(range(num_files)):
        temp = path_output.joinpath(regex.replace('*', f"{ii:06d}"))
        exists = temp.exists()
        if not exists:
            all_exist = False
            break

    if not all_exist:
        err = f"Missing at least file number {ii} out of {num_files=}!"
        log.exception(err)
        raise ValueError(err)

    # ---- Check one example data file
    temp = files[0]
    data = np.load(temp, allow_pickle=True)
    log.info(f"Test file: {temp=}\n\t{list(data.keys())=}")
    fobs = data['fobs']
    fobs_edges = data['fobs_edges']
    nfreqs = fobs.size
    temp_gwb = data['gwb'][:]
    nreals = temp_gwb.shape[1]
    test_params = data['params']
    param_names = data['names']
    lhs_grid = data['lhs_grid']
    try:
        pdim = data['pdim']
    except KeyError:
        pdim = 6

    try:
        nsamples = data['nsamples']
        if num_files != nsamples:
            raise ValueError(f"{nsamples=} but {num_files=} !!")
    except KeyError:
        pass

    assert np.ndim(temp_gwb) == 2
    if temp_gwb.shape[0] != nfreqs:
        raise ValueError(f"{temp_gwb.shape=} but {nfreqs=}!!")
    if temp_gwb.shape[1] != nreals:
        raise ValueError(f"{temp_gwb.shape=} but {nreals=}!!")

    # ---- Store results from all files

    gwb_shape = [num_files, nfreqs, nreals]
    shape_names = list(param_names[:]) + ['freqs', 'reals']
    gwb = np.zeros(gwb_shape)
    params = np.zeros((num_files, pdim))
    grid_idx = np.zeros((num_files, pdim), dtype=int)

    log.info(f"Collecting data from {len(files)} files")
    for ii, file in enumerate(tqdm.tqdm(files)):
        temp = np.load(file, allow_pickle=True)
        assert ii == temp['pnum']
        assert np.allclose(fobs, temp['fobs'])
        assert np.allclose(fobs_edges, temp['fobs_edges'])
        pars = [temp[nn][()] for nn in param_names]
        for jj, (pp, nn) in enumerate(zip(temp['params'], temp['names'])):
            assert np.allclose(pp, test_params[jj])
            assert nn == param_names[jj]

        assert np.all(lhs_grid == temp['lhs_grid'])

        tt = temp['gwb'][:]
        assert np.shape(tt) == (nfreqs, nreals)
        gwb[ii] = tt
        params[ii, :] = pars
        grid_idx[ii, :] = temp['lhs_grid_idx']
        if debug:
            break

    out_filename = path_output.joinpath('sam_lib.hdf5')
    log.info("Writing collected data to file {out_filename}")
    with h5py.File(out_filename, 'w') as h5:
        h5.create_dataset('fobs', data=fobs)
        h5.create_dataset('fobs_edges', data=fobs_edges)
        h5.create_dataset('gwb', data=gwb)
        h5.create_dataset('params', data=params)
        h5.create_dataset('lhs_grid', data=lhs_grid)
        h5.create_dataset('lhs_grid_indices', data=grid_idx)
        h5.attrs['param_names'] = np.array(param_names).astype('S')
        h5.attrs['shape_names'] = np.array(shape_names).astype('S')

    log.warning(f"Saved to {out_filename}, size: {holo.utils.get_file_size(out_filename)}")
    return

