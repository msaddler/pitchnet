import sys
import os
import h5py
import dask.array
import numpy as np
import glob
import pdb


def main(hdf5_regex, output_fn=None, write_mode='w'):
    '''
    Helper function for combining identically structured hdf5 files
    '''
    # Get the list of hdf5 files to merge
    hdf5_fn_list = sorted(glob.glob(hdf5_regex))
    print('HDF5 FILE LIST:')
    for fn in hdf5_fn_list: print('---|', fn)
    
    # Design an output filename for the merged hdf5 file
    if output_fn is None:
        output_fn = ''
        for char_idx, char in enumerate(hdf5_fn_list[0]):
            if np.all([char == fn[char_idx] for fn in hdf5_fn_list]):
                output_fn += char
        assert not output_fn in hdf5_fn_list, "output_fn must not be in hdf5_fn_list"
        print('OUTPUT FILENAME:')
        print('---|', output_fn)
    
    # Use the first file in hdf5_fn_list to get the paths to all the h5py.Dataset objects
    f = h5py.File(hdf5_fn_list[0], 'r')
    hdf5_dataset_key_list = []
    def get_dataset_paths(name, node):
        if isinstance(node, h5py.Dataset):
            hdf5_dataset_key_list.append(name)
    f.visititems(get_dataset_paths)
    print('INPUT HDF5 FILE STRUCTURE:')
    for key in hdf5_dataset_key_list: print('---|', key, f[key].shape, f[key].dtype)
    
    # Sort datasets into maindata (concatenated) and metadata (copied once)
    maindata_key_list = []
    metadata_key_list = []
    for key in hdf5_dataset_key_list:
        if f[key].shape[0] > 1: maindata_key_list.append(key)
        else: metadata_key_list.append(key)
    # Concatenate the maindata datasets into single dask arrays
    maindata_dask_arrays = {}
    for key in maindata_key_list:
        datasets_list = [h5py.File(fn)[key] for fn in hdf5_fn_list]
        dask_arrays_list = [dask.array.from_array(dataset, chunks=dataset.shape) for dataset in datasets_list]
        print('... concatenating `{}` dataset from {} files'.format(key, len(datasets_list)))
        maindata_dask_arrays[key] = dask.array.concatenate(dask_arrays_list, axis=0)
    # Copy the metadata datasets from the first hdf5 file into dask array
    metadata_dask_arrays = {}
    for key in metadata_key_list:
        print('... copying `{}` dataset from {}'.format(key, hdf5_fn_list[0]))
        metadata_dask_arrays[key] = dask.array.from_array(f[key], chunks=f[key].shape)
    
    # Generate new hdf5 file for the merged datasets
    print('[INITIALIZING] {}'.format(output_fn))
    f_out = h5py.File(output_fn, write_mode)
    if maindata_dask_arrays:
        print('[WRITING] maindata_dask_arrays to {}'.format(output_fn))
        for key in maindata_dask_arrays.keys():
            if maindata_dask_arrays[key].dtype.kind is 'O':
                # Special case of ragged Dask array --> hdf5 dtype needs to be made explicit
                dask_dset = maindata_dask_arrays[key]
                np_dtype = maindata_dask_arrays[key][0].compute().dtype
                hdf5_dtype = h5py.special_dtype(vlen=np_dtype)
                f_out.create_dataset(key, dask_dset.shape, dtype=h5py.special_dtype(vlen=np_dtype))
                print('---|', key, dask_dset.shape, dask_dset.dtype, np_dtype, hdf5_dtype)
                dask.array.to_hdf5(output_fn, {key: maindata_dask_arrays[key].astype(hdf5_dtype)})
            else:
                print('---|', key, maindata_dask_arrays[key].shape, maindata_dask_arrays[key].dtype)
                dask.array.to_hdf5(output_fn, {key: maindata_dask_arrays[key]})
    if metadata_dask_arrays:
        print('[WRITING] metadata_dask_arrays to {}'.format(output_fn))
        dask.array.to_hdf5(output_fn, metadata_dask_arrays)
    f.close()
    f_out.close()
    
    # Print the output file structure
    f_out = h5py.File(output_fn, 'r')
    hdf5_dataset_key_list = []
    f_out.visititems(get_dataset_paths)
    print('OUTPUT HDF5 FILE STRUCTURE:')
    for key in hdf5_dataset_key_list: print('---|', key, f_out[key].shape, f_out[key].dtype)
    f_out.close()
    print('[END] {}'.format(output_fn))


if __name__ == "__main__":
    ''' TEMPORARY COMMAND LINE USAGE '''
    assert_msg = "scipt usage: python <script_name> <hdf5_regex> <output_fn (OPTIONAL)>"
    assert len(sys.argv) == 2 or len(sys.argv) == 3, assert_msg
    hdf5_regex = str(sys.argv[1])
    output_fn = None
    if len(sys.argv) == 3: output_fn = str(sys.argv[2])
    main(hdf5_regex, output_fn=output_fn)
