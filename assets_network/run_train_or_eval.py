import sys
sys.path.append('/code_location/multi_gpu')
import os
import argparse
import glob
import json
import numpy as np
import model_routines_train_and_eval as routines
import functions_parameter_handling as fph
import rand_brain_arch_generator
import warnings
import pdb


def run_train_or_eval(output_directory,
                      train_regex,
                      eval_regex,
                      config_filename=None,
                      eval_only_mode=False,
                      force_overwrite=False,
                      eval_brain_ckpt=None,
                      eval_frontend_ckpt=None,
                      eval_output_fn='EVAL.json',
                      write_audio_files=False,
                      write_probs_out=False,
                      migrate_config=True):
    """
    Runs the training or evaluation for the IBM hearing aid code

    Args:
    -----
    output_directory (str): location for the saved models, where checkpoints will be saved and config file will be copied
    train_regex (str): regex for the location of tfrecords for the training dataset
    eval_regex (str): regex for the location of tfrecords for the validation or evaluation dataset
    config_filename (str): if specified, loads in the configuration from the given path
    eval_only_mode (boolean): if true, runs evaluation-only routine (instead of train routine with intermittent evaluation)
    force_overwrite (boolean): if specified, force overwrite the config file
    eval_brain_ckpt (int): can be used to specify a brain network checkpoint number in evaluation-only routine
    eval_frontend_ckpt (int): can be used to specify a frontend checkpoint number in evaluation-only routine
    eval_output_fn (str): filename for writing evaluation-only routine outputs to a JSON file
    write_audio_files (bool): if true, evaluation-only routine will write input and frontend (if applicable) audio files
    write_probs_out (bool): if true, evaluation-only routine will write final activations to file for all tasks
    """
    # Call os.path.normpath to standardize how the output_directory string is saved in config files
    output_directory = os.path.normpath(output_directory)

    # Make the output_directory if it does not exist
    if not os.path.exists(output_directory): os.makedirs(output_directory)

    # If no config file is specified, assume the config file is output_directory/config.json
    if config_filename is None:
        config_filename = os.path.join(output_directory, 'config.json')

    # Load the config file (NOTE the different behaviors for evaluation and training mode)
    if eval_only_mode:
        print('Loading CONFIG using `eval_only_mode=True` -- no config files are written or modified')
        CONFIG = fph.load_config_dict_from_json(config_filename)
        print('Loaded CONFIG from {}'.format(config_filename))
    else:
        if migrate_config:
            print('Loading CONFIG using `eval_only_mode=False` and  `migrate_config=True`')
            # Default behavior ensures config file is migrated to output_directory and does not rely on parameters
            # outside of output_directory (function will copy all parameter files to output_directory and overwrite
            # external filenames in config file).
            config_filename_to_load = fph.migrate_config_to_new_output_directory(config_filename, output_directory,
                                                                                 force_overwrite=force_overwrite)
        else:
            print('Loading CONFIG using `eval_only_mode=False` and  `migrate_config=False`')
            config_filename_to_load = config_filename
        # Load CONFIG (nested dictionary containing all parameters to run routine) from the migrated config file.
        CONFIG = fph.load_config_dict_from_json(config_filename_to_load)
        print('Loaded CONFIG from {}'.format(config_filename_to_load))


    # ===================================================================================================== #
    # If not eval_only_mode, run TRAINING ROUTINE (possibly with intermittent evaluation on validation set) #
    # ===================================================================================================== #
    if not eval_only_mode:
        # Add train_regex to CONFIG/EXTRAS for record-keeping purposes (EXTRAS are not used by the routine)
        if CONFIG.get('FRONTEND_PARAMS',{}): extra_regex_key = 'train_regex_frontend'
        else: extra_regex_key = 'train_regex_brain'
        CONFIG['EXTRAS'][extra_regex_key] = train_regex
        # Write this addition to the config file
        fph.write_config_dict_to_json(CONFIG, config_filename_to_load, force_overwrite=False,
                                      ignore_config_dict_keys=['EXTRAS'])

        ###### TEMPORARY PATCH FOR DEPRECATED `task_loss_weights` (2019JUL22)
        if 'task_loss_weights' in CONFIG.keys():
            assert not 'TASK_LOSS_PARAMS' in CONFIG.keys(), "both `task_loss_weights` and `TASK_LOSS_PARAMS` were specified"
            warnings.warn("HOT PATCHING USE OF DEPRECATED ARGUMENT: `task_loss_weights` (replaced by `TASK_LOSS_PARAMS`)")
            CONFIG['TASK_LOSS_PARAMS'] = {}
            for key in CONFIG['task_loss_weights']:
                CONFIG['TASK_LOSS_PARAMS'][key] = {'weight': CONFIG['task_loss_weights'][key]}

        if 'learning_rate' in CONFIG.keys():
            lr_warning = "learning_rate is defined both OPTM_PARAMS and top level CONFIG"
            assert not 'learning_rate' in CONFIG.get('OPTM_PARAMS', {}).keys(), lr_warning
            warnings.warn("HOT PATCHING USE OF DEPRECATED ARGUMENT: moving CONFIG['learning_rate'] "
                          "to CONFIG['OPTM_PARAMS']['learning_rate']")
            if 'OPTM_PARAMS' not in CONFIG.keys():
                CONFIG['OPTM_PARAMS'] = {}
            CONFIG['OPTM_PARAMS']['learning_rate'] = CONFIG['learning_rate']
            del CONFIG['learning_rate']
        ###### THIS PATCH WILL BE REMOVED SOON

        # Fill-in CONFIG default values for missing/abridged values in config file
        # (NOTE these changes are NOT stored in the config file)
        CONFIG = fph.fill_in_config_defaults(CONFIG, output_directory)

        # Print the final CONFIG parameters and run the routine
        print(fph.dict_to_formatted_string(CONFIG))
        routines.run_training_routine(train_regex, valid_regex=eval_regex, **CONFIG)


    # ===================================================================================================== #
    # If eval_only_mode, run the EVALUATION ROUTINE (only the forward-pass of the graph is constructed)     #
    # ===================================================================================================== #
    else:
        # TODO: we should work on how to provide all these parameters w/o call script)
        # TODO: since gradients are never built in this routine, do we actually have to set these `trainable` flags?

        # Modify CONFIG values for evaluation (these will be saved in the evaluation output file)
        if CONFIG.get('FRONTEND_PARAMS',{}):
            CONFIG['FRONTEND_PARAMS']['trainable'] = False
        CONFIG['BRAIN_PARAMS']['trainable'] = False
        CONFIG['BRAIN_PARAMS']['batchnorm_flag'] = False
        CONFIG['BRAIN_PARAMS']['dropout_flag'] = False

        # Search through the `feature_parsing_dict` and collect all of the metadata keys
        metadata_keys = [] # Metadata keys are keys in the tfrecords that are not signal or label
        for key in CONFIG['ITERATOR_PARAMS']['feature_parsing_dict'].keys():
            key_is_signal_key = len(CONFIG['ITERATOR_PARAMS']['feature_parsing_dict'][key].get('shape', [])) > 0
            key_is_label_key = key in CONFIG['N_CLASSES_DICT'].keys()
            if (not key_is_signal_key) and (not key_is_label_key): metadata_keys.append(key)
        # Setup the `audio_writing_params` (stays empty if not saving audio files)
        audio_writing_params = {}
        if write_audio_files:
            # TODO: these could be passed in via the CONFIG file, but it might be less hassle to rely on hard-coded defaults
            audio_writing_params = {
                'example_limit': 26, # Cap the number of stimuli to write to file
                'rms': 0.02, # If `rms` is None, False, or not specified, saved audio is not rms normalized
                'input_audio': 'saved_audio/input_audio_stimulus{:03}.wav', # path relative to output_directory
                'frontend_audio': 'saved_audio/frontend_audio_stimulus{:03}.wav', # path relative to output_directory
            }
        # Parameters to run the routine
        routine_params = {
            'output_directory': output_directory,
            'brain_net_ckpt_to_load': eval_brain_ckpt,
            'frontend_ckpt_to_load': eval_frontend_ckpt,
            'debug_print': False,
            'controller': '/cpu:0',
            'maindata_keyparts': [':labels_true', ':labels_pred', ':correct_pred'],
            'metadata_keys': metadata_keys,
            'audio_writing_params': audio_writing_params,
        }
        if write_probs_out: routine_params['maindata_keyparts'].append(':probs_out')
        # Routine parameters are added to CONFIG with no nesting
        for key in sorted(routine_params.keys()):
            CONFIG[key] = routine_params[key]

        # Run the evaluation-only routine
        print(fph.dict_to_formatted_string(CONFIG))
        results_dict = routines.run_eval_routine(eval_regex, **CONFIG)

        # Add the CONFIG dictionary used for evaluation to the results_dict
        results_dict['CONFIG'] = CONFIG

        # Remove large arrays from the results_dict and store separately as .npy files
        large_array_keys = [key for key in results_dict.keys() if 'probs_out' in key]
        for key in large_array_keys:
            large_array_result = np.array(results_dict.pop(key))
            fn_suffix = '_' + key.replace('/', '_').replace(':', '_') + '.npy'
            suffix_to_replace = eval_output_fn[eval_output_fn.rfind('.'):]
            large_array_fn = os.path.basename(eval_output_fn).replace(suffix_to_replace, fn_suffix)
            results_dict[key] = large_array_fn
            print('[WRITING] results_dict[`{}`] to {} (shape: {})'.format(
                key, os.path.join(output_directory, large_array_fn), large_array_result.shape))
            np.save(os.path.join(output_directory, large_array_fn), large_array_result)

        # Write results_dict to a JSON file
        # TODO: in the future, if we want to evaluate on large datasets and store large variables
        # we should find a more appropriate output format and perhaps write during evaluation.
        eval_output_fn = os.path.join(output_directory, eval_output_fn)
        print('[WRITING] evaluation results_dict to {}'.format(eval_output_fn))
        with open(eval_output_fn, 'w') as f: json.dump(results_dict, f)
        print('[END] wrote evaluation results_dict to {}'.format(eval_output_fn))

        print('File structure: {}'.format(eval_output_fn))
        for key in results_dict.keys():
            if type(results_dict[key]) == list: print('|___', key, np.array(results_dict[key]).shape)
            else: print('|___', key, results_dict[key], type(results_dict[key]))



if __name__ == "__main__":
    '''
    This script is designed to run from the command line. At least two arguments are
    required, an output_directory and a routine flag (-t and/or -e). Example usage:

    Training routine only:
        python run_train_or_eval.py <output_directory> -t
        python run_train_or_eval.py <output_directory> -t -c <config_file>
        python run_train_or_eval.py <output_directory> -t -c <config_file> -dt <train_regex>
    Evaluation routine only:
        python run_train_or_eval.py <output_directory> -e
        python run_train_or_eval.py <output_directory> -e -c <config_file>
        python run_train_or_eval.py <output_directory> -e -c <config_file> -de <eval_regex>
    Training with intermittent evaluation on a validation set:
        python run_train_or_eval.py <output_directory> -t -e
        python run_train_or_eval.py <output_directory> -t -e -c <config_file>
        python run_train_or_eval.py <output_directory> -t -e -c <config_file> -dt <train_regex> -de <valid_regex>
    '''
    parser = argparse.ArgumentParser(description="Run training pipeline")
    # Arguments for output directory and handling config files
    parser.add_argument('outputdir', type=str,
                        help='output directory for model training')
    parser.add_argument('-c', '--configfile', type=str,
                        help='location of the config file to use for the model (default is to look in output directory)')
    parser.add_argument('-f', '--force_overwrite', action='store_true', default=False,
                        help='if specified, force overwrite the config file')
    # Arguments to specify the tfrecords used for model training and evaluation
    parser.add_argument('-dt', '--tfrecordsregextrain', type=str, default='/data/train*/tfrecords/*.tfrecords',
                        help='regex that globs tfrecords for model training data')
    parser.add_argument('-de', '--tfrecordsregexeval', type=str, default='/data/valid*/tfrecords/*.tfrecords',
                        help='regex that globs tfrecords for model evaluation data')
    # Flags to specify whether model should run in training and/or evaluation mode
    parser.add_argument('-t', '--training', action='store_true', default=False,
                        help='if specified, model will be trained on tfrecordsregextrain')
    parser.add_argument('-e', '--evaluation', action='store_true', default=False,
                        help='if specified, model will be evaluated on tfrecordsregexeval')
    parser.add_argument('-ebc', '--eval_brain_ckpt', type=int, default=None,
                        help='if specified, load the specified brain net ckpt number instead of the most recent one')
    parser.add_argument('-efc', '--eval_frontend_ckpt', type=int, default=None,
                        help='if specified, load the specified wave net ckpt number instead of the most recent one')
    parser.add_argument('-efn', '--eval_output_fn', type=str, default='EVAL.json',
                        help='JSON filename to store evaluation outputs (eval_only_mode), do not include path')
    parser.add_argument('-waf', '--write_audio_files', action='store_true', default=False,
                        help='if specified, input and frontend audio files will be written (eval_only_mode)')
    parser.add_argument('-wpo', '--write_probs_out', action='store_true', default=False,
                        help='if specified, final activations will be written to file (eval_only_mode)')
    parser.add_argument('-mc', '--migrate_config', type=int, default=1,
                        help='if migrate_config is True, config file will be migrated to output_directory')
    args = parser.parse_args()

    # The specified combination of `-t` and `-e` flags determines what routines are run
    if (not args.training) and (args.evaluation):
        print('====== Evaluation Only Mode ======') # only `-e` specified
        eval_only_mode = True
    elif (args.training) and (not args.evaluation):
        print('====== Training Only Mode ======') # only `-t` specified
        eval_only_mode = False
    elif (args.training) and (args.evaluation):
        print('====== Training + Validation Mode ======') # both `-t` and `-e` specified
        eval_only_mode = False
    elif (not args.training) and (not args.evaluation):
        no_routine_msg = ('Please specify `-t` for training and/or `-e` for evaluation '
                          '(specifying both will run training with intermittent validation)')
        raise ValueError(no_routine_msg) # neither `-t` nor `-e` specified

    # Training and evaluation regex's are only passed on if their routine flags are indicated
    train_regex, eval_regex = (None, None)
    if args.training: train_regex = args.tfrecordsregextrain
    if args.evaluation: eval_regex = args.tfrecordsregexeval
    
    print('DATA_TRAIN:', train_regex)
    print('DATA_EVAL:', eval_regex)
    print('OUTPUT_DIR:', args.outputdir)
    print('CONFIG:', args.configfile)
    print('EVALUATION-ONLY MODE:', eval_only_mode)
    if eval_only_mode: print('EVALUATION OUTPUT:', os.path.join(args.outputdir, args.eval_output_fn))

    run_train_or_eval(args.outputdir, train_regex, eval_regex,
                      config_filename=args.configfile, eval_only_mode=eval_only_mode,
                      force_overwrite=args.force_overwrite,
                      eval_brain_ckpt=args.eval_brain_ckpt,
                      eval_frontend_ckpt=args.eval_frontend_ckpt,
                      eval_output_fn=args.eval_output_fn,
                      write_audio_files=args.write_audio_files,
                      write_probs_out=args.write_probs_out,
                      migrate_config=bool(args.migrate_config))
