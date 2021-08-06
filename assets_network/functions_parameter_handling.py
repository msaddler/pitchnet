import os
import json
import shutil
import filecmp
import copy
from functools import reduce
import operator


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


def get_n_classes_dict(task_names):
    '''
    This function accepts a list of task names and returns a dictionary mapping
    tfrecords label paths to numbers of classes in the dataset.
    
    Args:
        task_names (string or list): valid tasks are word, speaker, audioset
    Returns:
        N_CLASSES_DICT (dict): keys are tfrecords label paths, values are numbers of classes
    Raises:
        ValueError: If a string in task_names does not correspond to a valid task
    '''
    if type(task_names) is str: task_names = [task_names]
    N_CLASSES_DICT = {}
    for task_name in task_names:
        print('>>> Including parameters for task: {} <<<'.format(task_name))
        if task_name.lower() == 'word':
            feature_label_path = '/stimuli/word_int' # tfrecords path for word labels
            n_classes = 830
        elif task_name.lower() == 'speaker':
            feature_label_path = '/stimuli/speaker_int' # tfrecords path for speaker labels
            n_classes = 442
        elif task_name.lower() == 'audioset':
            feature_label_path = '/stimuli/labels_binary_via_int' # tfrecords path for audioset labels
            n_classes = 519
        else:
            raise ValueError('TASK NOT RECOGNIZED: {}'.format(task_name.lower()))
        N_CLASSES_DICT[feature_label_path] = n_classes
    return N_CLASSES_DICT


def get_iterator_params(N_CLASSES_DICT, feature_signal_path='stimuli/signal', signal_shape=(40000,)):
    '''
    This function accepts `N_CLASSES_DICT` from the previous function and returns
    the dictionary `ITERATOR_PARAMS` (`N_CLASSES_DICT` determines the labels).
    
    Args:
        N_CLASSES_DICT (dict): keys are tfrecords label paths, values are numbers of classes
        feature_signal_path (string): tfrecords path to the input waveforms
        signal_shape (tuple): shape of waveforms stored in tfrecords ((40000,) for 2s stimuli at 20kHz)
    Returns:
        ITERATOR_PARAMS (dict): parameter dictionary for `build_input_iterator` function
            NOTE: `feature_parsing_dict` is nested within `ITERATOR_PARAMS`
    '''
    ITERATOR_PARAMS = {}
    ITERATOR_PARAMS['feature_signal_path'] = feature_signal_path
    feature_parsing_dict = {}
    feature_parsing_dict[feature_signal_path] = {'dtype':'tf.float64', 'shape':signal_shape,}
    for key in N_CLASSES_DICT:
        if 'labels_binary_via_int' in key:
            feature_parsing_dict[key] = {'dtype':'tf.int64', 'shape': (N_CLASSES_DICT[key],),}
        else:
            feature_parsing_dict[key] = {'dtype': 'tf.int64',}
    ITERATOR_PARAMS['feature_parsing_dict'] = feature_parsing_dict
    return ITERATOR_PARAMS


def get_reasonable_coch_params():
    COCH_PARAMS = {
        'compression': 'stable_point3',
        'return_subbands_only': True,
        'rectify_and_lowpass_subbands': True,
        'rFFT': True,
        'N': 40,
        'filter_type': 'roex',
        'min_cf': 50,
        'max_cf': 8e3,
        'bandwidth_scale_factor': 1.0,
    }
    return COCH_PARAMS


def get_reasonable_brain_params(output_directory, arch_config_fn='config_brain_net_arch.json',
                                pckl_fn='pckl_brain_net.pckl'):
    assert os.path.basename(arch_config_fn) == arch_config_fn, 'arch_config_fn must not contain a path'
    assert os.path.basename(pckl_fn) == pckl_fn, 'pckl_fn must not contain a path'
    BRAIN_PARAMS = {
        'trainable': True, 
        'config': os.path.join(output_directory, arch_config_fn), 
        'batchnorm_flag': True,
        'dropout_flag': True,
        'save_pckl_path': os.path.join(output_directory, pckl_fn),
    }
    return BRAIN_PARAMS


def get_reasonable_wavenet_params():
    WAVENET_PARAMS = {
        'trainable': True,
        'padding': [[0,0], [4093,4093]],
        'USE_FP16': False,
        'LEN_OUTPUT': 40000, # THIS ASSUMES SAMPLING RATE IS 20kHZ AND DURATION IS 2s
        'NUM_BLOCKS_CLEAN': 4, 
        'NUM_LAYERS_CLEAN': 10, 
        'NUM_BLOCKS_NOISY': 4, 
        'NUM_LAYERS_NOISY': 10, 
        'NUM_CLASSES': 256, 
        'NUM_POST_LAYERS': 2, 
        'NUM_RESIDUAL_CHANNELS_CLEAN': 64, 
        'NUM_RESIDUAL_CHANNELS_NOISY': 64, 
        'NUM_SKIP_CHANNELS': 256,
    }
    return WAVENET_PARAMS


def get_reasonable_config(output_directory, tfrecords_regex, task_names=['word'],
                          arch_config_fn='test_arch_hpool_batchnorm_subbands_multi_fc.json'):
    '''
    This function generates an example CONFIG dictionary. This function does not read
    or write any files.
    
    Args:
        output_directory (string): specifies directory where all model checkpoints and parameters will be saved
        tfrecords_regex (string): regex that globs tfrecords data
        task_names (list of strings): specifies which tasks to include ('word', 'speaker', 'audioset')
        arch_config_fn (string): JSON filename used to save brain net architecture config file in the output directory
    Returns:
        CONFIG (dict): example dictionary containing all parameters to run brain net training 
    '''
    # Make the default config_filename
    config_filename = os.path.join(output_directory, 'config.json')
    
    # Setup the task-specific parameters
    N_CLASSES_DICT = get_n_classes_dict(task_names)
    ITERATOR_PARAMS = get_iterator_params(N_CLASSES_DICT,
                                          feature_signal_path='stimuli/signal',
                                          signal_shape=(40000,))

    # Wavenet parameters (empty dict means wavenet will not be included in graph)
    WAVENET_PARAMS = {}

    # Peripheral model parameters
    COCH_PARAMS = get_reasonable_coch_params()

    # Brain network parameters
    BRAIN_PARAMS = get_reasonable_brain_params(output_directory, arch_config_fn=arch_config_fn)

    # Optional EXTRAS (not used by the routine, but are stored in the config file)
    EXTRAS = {
        'task_names': task_names,
        'tfrecords_regex': tfrecords_regex,
        'config_filename': config_filename,
    }

    # Merge parameter dictionaries into a nested CONFIG dictionary
    CONFIG = create_config_dict_from_kwargs(N_CLASSES_DICT=N_CLASSES_DICT,
                                            ITERATOR_PARAMS=ITERATOR_PARAMS,
                                            COCH_PARAMS=COCH_PARAMS,
                                            WAVENET_PARAMS=WAVENET_PARAMS,
                                            BRAIN_PARAMS=BRAIN_PARAMS,
                                            EXTRAS=EXTRAS)

    # Parameters to run the routine
    ROUTINE_PARAMS = {
        'output_directory': output_directory,
        'brain_net_ckpt_to_load': None,
        'wavenet_ckpt_to_load': None,
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 1000,
        'display_step': 5,
        'save_step': 1000,
        'debug_print': False,
        'controller': '/cpu:0',
    }
    # Routine parameters are added to CONFIG with no nesting
    for key in sorted(ROUTINE_PARAMS.keys()):
        CONFIG[key] = ROUTINE_PARAMS[key]
        
    return CONFIG


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
    if "MULTIBRAIN_PARAMS" in CONFIG['BRAIN_PARAMS'].keys():
        CONFIG, ignore_config_dict_keys = migrate_brain_config_multibrain(CONFIG,new_output_directory,force_overwrite)
    else:
        CONFIG, ignore_config_dict_keys = migrate_brain_config_standard(CONFIG,new_output_directory,force_overwrite)
    
    # Change `config_filename` in CONFIG and write CONFIG to the new `new_config_filename`
    new_config_filename = os.path.join(new_output_directory, 'config.json')
    if not CONFIG['EXTRAS'].get('config_filename', '') == new_config_filename:
        CONFIG['EXTRAS']['config_filename'] = new_config_filename
    write_config_dict_to_json(CONFIG, new_config_filename, force_overwrite=force_overwrite,
                              ignore_config_dict_keys=ignore_config_dict_keys)
    return new_config_filename


def migrate_brain_config_multibrain(CONFIG,new_output_directory,force_overwrite):
    # Copy all of the files named in BRAIN_PARAMS to the new_output_directory
    brain_param_keys_to_copy = ['config']
    brain_param_keys_to_rename = ['save_pckl_path', 'save_arch_path', 'save_ckpt_path']
    # keys to ignore in the config similarity check
    ignore_config_dict_keys = []
    for arch_key in CONFIG['BRAIN_PARAMS']['MULTIBRAIN_PARAMS'].keys():
        for key in brain_param_keys_to_copy + brain_param_keys_to_rename:
            if key in CONFIG['BRAIN_PARAMS']['MULTIBRAIN_PARAMS'][arch_key].keys():
                tmp = CONFIG['BRAIN_PARAMS']['MULTIBRAIN_PARAMS'][arch_key][key]
                tmpdir, tmpbasename = os.path.split(tmp)
                tmp_dest_filename = os.path.join(new_output_directory, tmpbasename)
                if not tmp_dest_filename == tmp:
                    # Change filenames in CONFIG to point to new_output_directory
                    print('Renaming in CONFIG: {} --> {}'.format(tmp, tmp_dest_filename))
                    CONFIG['BRAIN_PARAMS']['MULTIBRAIN_PARAMS'][arch_key][key] = tmp_dest_filename
                    if os.path.isfile(tmp) and (key in brain_param_keys_to_copy):
                        # For files that already exist, copy them to the new output directory
                        print('Copying files: {} --> {}'.format(tmp, tmp_dest_filename))
                        if (not force_overwrite) and (os.path.isfile(tmp_dest_filename)):
                            msg = ('destination file {} exists and differs from BRAIN_PARAMS[{}] '
                                   'file {} (override with `force_overwrite=True`, DANGEROUS!)')
                            assert filecmp.cmp(tmp_dest_filename, tmp, shallow=False), msg.format(tmp_dest_filename, key, tmp)
                        else: 
                            shutil.copyfile(tmp, tmp_dest_filename)
                ignore_config_dict_keys.append('BRAIN_PARAMS/MULTIBRAIN_PARAMS'+'/'+ 
                                               arch_key+'/'+key)
    return CONFIG, ignore_config_dict_keys


def migrate_brain_config_standard(CONFIG,new_output_directory,force_overwrite):
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


def fill_in_config_defaults(CONFIG, output_directory):
    '''
    This function helps make config files easier to work with by filling
    in reasonable defaults for missing or abridged values in CONFIG just
    prior to running the training routine and after migrating the config
    file to the output directory.
    
    Args
    ----
    CONFIG (dict): dictionary containing all parameters to run training routine
    output_directory (str): path to output directory for current model
    
    Returns
    -------
    CONFIG (dict): dictionary containing all parameters to run training routine
    '''
    # Fill-in defaults for: CONFIG['BRAIN_PARAMS']['MULTIBRAIN_PARAMS']
    if 'MULTIBRAIN_PARAMS' in CONFIG['BRAIN_PARAMS'].keys():
        for brain_key in CONFIG['BRAIN_PARAMS']['MULTIBRAIN_PARAMS'].keys():
            default_values = {
                'config': os.path.join(output_directory, brain_key + '.json'),
                'save_arch_path': os.path.join(output_directory, brain_key + '.json'),
                'save_ckpt_path': os.path.join(output_directory, brain_key + '.ckpt'),
                'save_pckl_path': os.path.join(output_directory, brain_key + '.pckl'),
            }
            for key in default_values.keys():
                val = CONFIG['BRAIN_PARAMS']['MULTIBRAIN_PARAMS'][brain_key].get(key, None)
                if val is None:
                    # If no value is provided at all, use the default value
                    CONFIG['BRAIN_PARAMS']['MULTIBRAIN_PARAMS'][brain_key][key] = default_values[key]
                elif os.path.basename(val) == val:
                    # If a relative path is provided, append the output_directory to the beginning
                    CONFIG['BRAIN_PARAMS']['MULTIBRAIN_PARAMS'][brain_key][key] = os.path.join(output_directory, val)
    # TODO: add other defaults here
    return CONFIG


def get_from_dict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)

def remove_from_dict(dataDict, mapList):
    get_from_dict(dataDict, mapList[:-1]).pop(mapList[-1], None)
