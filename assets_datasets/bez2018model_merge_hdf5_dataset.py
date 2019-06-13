import h5py
import dask.array as da
import numpy as np
import glob


infn_pattern = '/om/user/msaddler/data_pitch50ms_bez2018/NRTW-jwss_trainset_CF50-SR70-sp2-cohc00_filt00_30-90dB_*.hdf5'
outfn = infn_pattern.replace('_*', '')
if 'trainset' in outfn:
    outfn = outfn.replace('trainset', 'train')
elif 'valset' in outfn:
    outfn = outfn.replace('valset', 'valid')
elif 'stimset' in outfn:
    outfn = outfn.replace('stimset', 'stim')
elif 'puretonesset' in outfn:
    outfn = outfn.replace('puretonesset', 'puretones')
else:
    outfn = outfn.replace('set', '')


# Gather .hdf5 files to merge
filenames = sorted(glob.glob(infn_pattern))
print(len(filenames))
print(filenames[0])
print(filenames[-1])
print(outfn)

# Open first .hdf5 file to access keys
f = h5py.File(filenames[0])

# Generate and concatenate lists of dask arrays for main datasets in all files
main_arrays = {}
for key in f.keys():
    if isinstance(f[key], h5py.Dataset):
        dsets_key = [h5py.File(fn)[key] for fn in filenames]
        key_chunk = dsets_key[0].shape
        main_arrays_list = [da.from_array(dset, chunks=key_chunk) for dset in dsets_key]
        key_str = key
        print('Concatenating {} datasets from {} files'.format(key_str, len(main_arrays_list)))
        main_arrays[key_str] = da.concatenate(main_arrays_list, axis=0)
        if key_str == 'meanrates':
            print('<> <> Note: casting meanrates to np.float32 <> <>')
            main_arrays[key_str] = main_arrays[key_str].astype(np.float32)

# Generate and concatenate lists of dask arrays for augmentation datasets in all files
aug_arrays = {}
if 'augmentation' in f.keys():
    for key in f['augmentation'].keys():
        dsets_key = [h5py.File(fn)['augmentation'][key] for fn in filenames]
        key_chunk = dsets_key[0].shape
        aug_arrays_list = [da.from_array(dset, chunks=key_chunk) for dset in dsets_key]
        key_str = '/augmentation/' + key
        print('Concatenating {} datasets from {} files'.format(key_str, len(aug_arrays_list)))
        aug_arrays[key_str] = da.concatenate(aug_arrays_list, axis=0)
        
# Generate and concatenate lists of dask arrays for diagnostic datasets in all files
diag_arrays = {}
if 'diagnostic' in f.keys():
    for key in f['diagnostic'].keys():
        dsets_key = [h5py.File(fn)['diagnostic'][key] for fn in filenames]
        key_chunk = dsets_key[0].shape
        diag_arrays_list = [da.from_array(dset, chunks=key_chunk) for dset in dsets_key]
        key_str = '/diagnostic/' + key
        print('Concatenating {} datasets from {} files'.format(key_str, len(diag_arrays_list)))
        diag_arrays[key_str] = da.concatenate(diag_arrays_list, axis=0)

# Generate new hdf5 file for the concatenated arrays
f_out = h5py.File(outfn, 'w-')

if main_arrays:
    print('WRITING: main_arrays')
    da.to_hdf5(outfn, main_arrays)
if aug_arrays:
    print('WRITING: aug_arrays')
    da.to_hdf5(outfn, aug_arrays)
if diag_arrays:
    print('WRITING: diag_arrays')
    da.to_hdf5(outfn, diag_arrays)

print('COPYING: config folder from first .hdf5 file to output .hdf5 file')
f.copy('/config', f_out)
for x in list(f_out['config'].values()):
    print('/config/', x, x[0])
for x in list(f_out.values()):
    print(x)

# Close the open datasets
f.close()
f_out.close()

print('END:', outfn)
