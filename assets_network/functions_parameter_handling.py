import os
import json
import shutil
import filecmp
import copy


def create_config_dict_from_kwargs(**kwargs):
    '''
    Compiles keyword arguments into a single dictionary. Intended use is
    for compiling all PARAMS dict into a single nested CONFIG dict.
    '''
    return kwargs


def write_config_dict_to_json(CONFIG, filename, force_overwrite=False, ignore_config_dict_keys=[]):
    '''
    This function writes a dictionary (`CONFIG`) to a formatted json file
    with the specified `filename`. If the `filename` already exists, this
    function will ensure that the data in `CONFIG` exactly matches that in
    the pre-existing file before overwriting.
    
    Args:
        CONFIG (dict): JSON-serialiazeable (nested) dictionary to save
        filename (string): full path for output JSON file
        force_overwrite (boolean, default=False): if set to True, this
            function will overwrite `filename` even if the data in `CONFIG`
            does not match data loaded from pre-existing `filename`
        ignore_config_dict_keys (list) : ignores any keys present in the list
            when checking to see if the config files are the same. 
    '''
    if os.path.isfile(filename):
        print('Pre-existing config file found: {}'.format(filename))
        assert_msg = 'CONFIG does not match existing config file (override with force_overwrite=True)'
        assert (config_dict_matches_config_filename(CONFIG, filename, ignore_config_dict_keys=ignore_config_dict_keys) or force_overwrite), assert_msg
    with open(filename, 'w') as f: json.dump(CONFIG, f, sort_keys=True, indent=4)
    print('Successfully (over)wrote config file: {}'.format(filename))


def load_config_dict_from_json(filename):
    assert os.path.isfile(filename), "Attempted to load nonexistent file {}".format(filename)
    with open(filename, 'r') as f: CONFIG_LOADED = json.load(f)
    return CONFIG_LOADED


def dict_to_formatted_string(d, sort_keys=True, indent=4):
    return json.dumps(d, sort_keys=sort_keys, indent=indent)


def are_identical_dicts(dict1, dict2):
    dict1_string = dict_to_formatted_string(dict1)
    dict2_string = dict_to_formatted_string(dict2)
    return hash(dict1_string) == hash(dict2_string)


def config_dict_matches_config_filename(config_dict, config_filename, ignore_config_dict_keys=[]):
    config_dict_reference = load_config_dict_from_json(config_filename) # Load reference dictionary from config_filename
    config_dict_query = copy.deepcopy(config_dict) # The query dictionary is a copy of the provided config_dict
    # Strip values corresponding to the `ignore_config_dict_keys` from both the query and the reference dictionaries
    print("config_dict similarity check is ignoring: {}".format(ignore_config_dict_keys))
    for k in ignore_config_dict_keys:
        k_nested = k.split('/')
        remove_from_dict(config_dict_query,k_nested)
        remove_from_dict(config_dict_reference,k_nested)
    # Return boolean indiciating if the query and the reference dictionaries are identical
    return are_identical_dicts(config_dict_query, config_dict_reference)    


def migrate_config_to_new_output_directory(config_filename, new_output_directory,
                                           force_overwrite=False):
    '''
    This function is for migrating a CONFIG dictionary to a new output directory.
    (0) Load the CONFIG dictionary from the specified `config_filename`
    (1) Change CONFIG['output_directory'] to the new output directory
    (2) Search for filenames in CONFIG['BRAIN_PARAMS'] and copy them to new output directory
    (3) Change the filenames in CONFIG['BRAIN_PARAMS'] to point to the newly copied files
    (4) Set CONFIG['EXTRAS']['config_filename'] to `<new_output_directory>/config.json`
    (5) Save the CONFIG dictionary to `<new_output_directory>/config.json`
    
    Note that this function currently only considers specific, hard-coded portions of the CONFIG
    dicionary. This function will need to be updated if we add more filenames to different
    portions of CONFIG (e.g. filenames in WAVENET_PARAMS)
    
    Args:
        config_filename (str): full path and filename for the source config.json (file is only read)
        new_output_directory (str): path for destination directory (files will be copied here)
        force_overwrite (bool): set to True to allow for overwriting of files in new_output_directory
    Returns:
        new_config_filename (str): full path and filename for the saved config.json
    '''
    # Load the CONFIG dictionary from config_filename
    CONFIG = load_config_dict_from_json(config_filename)
    
    # Change `output_directory` in CONFIG
    assert os.path.isdir(new_output_directory), 'new_output_directory is not a directory'
    CONFIG['output_directory'] = new_output_directory
    
    # Copy all of the files named in BRAIN_PARAMS to the new_output_directory
    CONFIG, ignore_config_dict_keys = migrate_brain_config_standard(CONFIG,new_output_directory,force_overwrite)
    
    # Change `config_filename` in CONFIG and write CONFIG to the new `new_config_filename`
    new_config_filename = os.path.join(new_output_directory, 'config.json')
    if not CONFIG['EXTRAS'].get('config_filename', '') == new_config_filename:
        CONFIG['EXTRAS']['config_filename'] = new_config_filename
    write_config_dict_to_json(CONFIG, new_config_filename, force_overwrite=force_overwrite,
                              ignore_config_dict_keys=ignore_config_dict_keys)
    return new_config_filename


def migrate_brain_config_standard(CONFIG, new_output_directory, force_overwrite):
    brain_param_keys_to_copy = ['config']
    brain_param_keys_to_rename = ['save_pckl_path', 'save_arch_path', 'save_ckpt_path']
    # keys to ignore in the config similarity check
    ignore_config_dict_keys = []
    for key in brain_param_keys_to_copy + brain_param_keys_to_rename:
        if key in CONFIG['BRAIN_PARAMS'].keys():
            tmp = CONFIG['BRAIN_PARAMS'][key]
            tmpdir, tmpbasename = os.path.split(tmp)
            tmp_dest_filename = os.path.join(new_output_directory, tmpbasename)
            if not tmp_dest_filename == tmp:
                # Change filenames in CONFIG to point to new_output_directory
                print('Renaming in CONFIG: {} --> {}'.format(tmp, tmp_dest_filename))
                CONFIG['BRAIN_PARAMS'][key] = tmp_dest_filename
                if os.path.isfile(tmp) and (key in brain_param_keys_to_copy):
                    # For files that already exist, copy them to the new output directory
                    print('Copying files: {} --> {}'.format(tmp, tmp_dest_filename))
                    if (not force_overwrite) and (os.path.isfile(tmp_dest_filename)):
                        msg = ('destination file {} exists and differs from BRAIN_PARAMS[{}] '
                               'file {} (override with `force_overwrite=True`, DANGEROUS!)')
                        assert filecmp.cmp(tmp_dest_filename, tmp, shallow=False), msg.format(tmp_dest_filename, key, tmp)
                    else: 
                        shutil.copyfile(tmp, tmp_dest_filename)
            ignore_config_dict_keys.append('BRAIN_PARAMS/' + key)
    return CONFIG, ignore_config_dict_keys
