import sys
import os
import json
import numpy as np
import tensorflow as tf

import arch_generate_random_CNN

sys.path.append('/code_location/multi_gpu/')
import functions_brain_network
import functions_parameter_handling


def generate_possible_architecture(input_shape=[None, 100, 500, 1], n_classes_dict={'label': 700}):
    '''
    '''
    impossible_architecture = True
    attempt_count = 0
    while impossible_architecture:
        network_layer_list, _ = arch_generate_random_CNN.get_random_cnn_architecture()
        try:
            tf.reset_default_graph()
            input_tensor = tf.placeholder(tf.float32, shape=input_shape, name='input_tensor')
            output_tensor, network_tensors = functions_brain_network.make_brain_net(input_tensor,
                                                                                    n_classes_dict,
                                                                                    network_layer_list,
                                                                                    trainable=True,
                                                                                    batchnorm_flag=True,
                                                                                    dropout_flag=True,
                                                                                    save_arch_path=None,
                                                                                    save_pckl_path=None,
                                                                                    only_include_layers=None)
            impossible_architecture = False
        except ValueError:
            attempt_count += 1
            pass
    return network_layer_list, network_tensors, attempt_count


def save_network_architecture(network_layer_list, network_arch_fn):
    '''
    '''
    # Define helper class to JSON serialize the results_dict
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.int64): return int(obj)  
            return json.JSONEncoder.default(self, obj)
    # Save network_layer_list to JSON file named network_arch_fn
    if '.json' not in network_arch_fn:
        network_arch_fn = network_arch_fn + '.json'
    with open(network_arch_fn, 'w') as f:
        json.dump(network_layer_list, f, cls=NumpyEncoder, sort_keys=True)
    return network_arch_fn


def generate_output_directory(output_dir, source_config_fn='config_arch_search_v01.json'):
    '''
    '''
    # Create output directory if it does not exist
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    # Migrate the source config file to the new output directory
    dest_config_fn = functions_parameter_handling.migrate_config_to_new_output_directory(
        source_config_fn, output_dir, force_overwrite=False)
    # Get the network architecture filename from the newly generated config file
    CONFIG = functions_parameter_handling.load_config_dict_from_json(dest_config_fn)
    network_arch_fn = CONFIG['BRAIN_PARAMS']['save_arch_path']
    # Generate a random architecture and save it to `network_arch_fn`
    feature_signal_path = CONFIG['ITERATOR_PARAMS']['feature_signal_path']
    input_shape = CONFIG['ITERATOR_PARAMS']['feature_parsing_dict'][feature_signal_path]['shape']
    input_shape = [None] + input_shape
    while len(input_shape) < 4: input_shape = input_shape + [1]
    n_classes_dict = CONFIG['N_CLASSES_DICT']
    network_layer_list, network_tensors, attempt_count = generate_possible_architecture(input_shape=input_shape,
                                                                                        n_classes_dict=n_classes_dict)
    saved_network_arch_fn = save_network_architecture(network_layer_list, network_arch_fn)
    assert saved_network_arch_fn == network_arch_fn, "network_arch_fn was changed"
    return dest_config_fn


def main():
    dest_config_fn = generate_output_directory(sys.argv[1])
    print(dest_config_fn)


if __name__ == "__main__":
    main()
