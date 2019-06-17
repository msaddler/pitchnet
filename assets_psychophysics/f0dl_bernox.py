import sys
import os
import json
import numpy as np

sys.path.append('../assets_datasets/')
import stimuli_f0_labels


def load_f0_expt_dict_from_json(json_fn,
                                f0_label_true_key='f0_label:labels_true',
                                f0_label_pred_key='f0_label:labels_pred',
                                metadata_key_list=['low_harm', 'upp_harm', 'phase_mode', 'f0']):
    '''
    '''
    # Load the entire json file as a dictionary
    with open(json_fn, 'r') as f: json_dict = json.load(f)
    # Return dict with only specified fields
    expt_dict = {}
    assert f0_label_true_key in json_dict.keys(), "f0_label_true_key not found in json file"
    assert f0_label_pred_key in json_dict.keys(), "f0_label_pred_key not found in json file"
    expt_dict = {
        'f0_label_true': np.array(json_dict[f0_label_true_key]),
        'f0_label_pred': np.array(json_dict[f0_label_pred_key]),
    }
    for key in metadata_key_list:
        assert key in json_dict.keys(), "metadata key `{}` not found in json file".format(key)
        if isinstance(json_dict[key], list): expt_dict[key] = np.array(json_dict[key])
        else: expt_dict[key] = json_dict[key]
    return expt_dict


def add_f0_estimates_to_expt_dict(expt_dict, **kwargs_f0_bins):
    '''
    '''
    bins = stimuli_f0_labels.get_f0_bins(**kwargs_f0_bins)
    if not 'f0' in expt_dict.keys():
        expt_dict['f0'] = stimuli_f0_labels.label_to_f0(expt_dict['f0_label_true'], bins)
    if not 'f0_pred' in expt_dict.keys():
        expt_dict['f0_pred'] = stimuli_f0_labels.label_to_f0(expt_dict['f0_label_pred'], bins)
    return expt_dict
    
    



def main_tmp():
    fn = '/om2/user/msaddler/pitchnet/saved_models/test_ibm0/EVAL.json'
    with open(fn, 'r') as outfile:
        output_dict = json.load(outfile)
    for key in output_dict.keys():
        print(key, type(output_dict[key]))
        if isinstance(output_dict[key], str) or isinstance(output_dict[key], int): print(output_dict[key])

if __name__ == "__main__":
    ''' TEMPORARY COMMAND-LINE USAGE '''
    main_tmp()