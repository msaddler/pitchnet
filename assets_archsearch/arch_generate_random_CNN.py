import numpy as np
import sys
import os 
import json


class RandomCNN: 
    '''
    Generates a random CNN architecture class by sampling from a set of priors.
    '''
    def __init__(self):
        
        # Fixed parameters (describe all randomly generated networks)
        self.poolmethod = "hpool"
        self.batch_normalization = True # batch normalization after all layers
        self.activation_type = 'tf.nn.relu'
        self.activation_name = 'relu'
        self.fc_batch_norm = True
        self.fc_activation = True
        self.conv_padding = 'VALID_TIME' # height (frequency) is zero-padded, length (time) is valid-padded
        self.pool_padding = 'VALID_TIME' # height (frequency) is zero-padded, length (time) is valid-padded
        self.early_stop = True
        self.dropout = True
        self.rate = 0.5
        self.dilation_rate = [1,1]
        self.include_classification_layer = True
        
        # Range from which to sample number of convolutional layers (inclusive)
        MAX_LAYERS=8
        self.range_num_conv_layers = [1, MAX_LAYERS] # [lower, upper]
        
        # Discrete priors over convolutional kernel heights (freq) and lengths (time) for each layer  
        self.possible_conv_layer_height = [[1, 2, 3, 4, 5]] * MAX_LAYERS
        self.possible_conv_layer_length = [
            [16,32,64,128],
            [8,16,32,64],
            [4,8,16,32],
            [2,4,8,16],
            [2,4,8,16],
            [1,2,3,4],
            [1,2,3,4],
            [1,2,3,4],
        ]
        # Discrete prior over number of convolutional kernels for each layer
        self.possible_conv_layer_nums = [
            [16,32,64],
            [16,32,64,128],
            [32,64,128,256],
            [32,64,128,256,512],
            [32,64,128,256,512],
            [32,64,128,256,512],
            [64,128,256,512],
            [64,128,256,512],
        ]
        
        # Discrete priors over convolutional strides for each layer
        self.possible_conv_strides_height = [[1]] * MAX_LAYERS # NO STRIDED CONVOLUTION
        self.possible_conv_strides_length = [[1]] * MAX_LAYERS # NO STRIDED CONVOLUTION
        
        # Discrete priors over pooling sizes after each conv layer ("1" pooling = no pooling)
        self.possible_pooling_strides_height = [
            [1,2,3],
            [1,2,3],
            [1,2],
            [1,2],
            [1,1,2],
            [1,1,2],
            [1,1,1,2],
            [1,1,1,2],
        ]
        self.possible_pooling_strides_length = [
            [2,4,8,16],
            [2,4,8],
            [1,2,4],
            [1,2],
            [1,2],
            [1,1,2],
            [1,1,2],
            [1,1,1,2],
        ]
        
        # Discrete prior over existence and size of an intermediate fully-connected layer
        self.possible_layer_fc=[False,True]
        self.possible_num_fc_units=[128,256,512,1024]
        
        ###### SAMPLE RANDOM ARCHITECTURE FROM THE PRIORS SET ABOVE ######
        
        # Choose the parameters for the conv layers
        self.num_conv_kernels, self.convstrides, self.poolstrides, self.conv_kernels_sizes = uniformly_sample_conv_layers(
            self.range_num_conv_layers,
            self.possible_conv_layer_height,
            self.possible_conv_layer_length,
            self.possible_conv_layer_nums,
            self.possible_conv_strides_height,
            self.possible_conv_strides_length,
            self.possible_pooling_strides_height,
            self.possible_pooling_strides_length)
        
        # Flags for two pooling methods (hard coded) 
        if self.poolmethod == "hpool":
            self.pool_size = [[x*4 if x > 1 else 1 for x in stride] for stride in self.poolstrides] 
            extra_pooling_args = {'normalize': True, 'sqrt_window': False}
        elif self.poolmethod == 'tf.layers.max_pooling2d':
            self.pool_size = self.poolstrides
            extra_pooling_args = {}
        
        # Add the parameters from the possible fully connected layer
        self.include_fc_intermediate = self.possible_layer_fc[np.random.randint(0, len(self.possible_layer_fc))]
        if self.include_fc_intermediate: 
            self.num_fc_units = self.possible_num_fc_units[np.random.randint(0, len(self.possible_num_fc_units))]
        
        ###### BUILD THE SAMPLED RANDOM ARCHITECTURE ######
        
        all_layer_list = []
        for layer_idx in range(len(self.num_conv_kernels)):
            # Convolutional layer
            conv_layer_dict = {
                'layer_type': 'tf.layers.conv2d',
                'args': {
                    'filters': self.num_conv_kernels[layer_idx],
                    'kernel_size': self.conv_kernels_sizes[layer_idx],
                    'padding': self.conv_padding,
                    'strides': self.convstrides[layer_idx],
                    'dilation_rate': self.dilation_rate,
                    'activation': None,
                    'name': 'conv_{:d}'.format(layer_idx),
                },
            }
            all_layer_list.append(conv_layer_dict)
            # Activation function
            activation_function_dict = {
                'layer_type': self.activation_type,
                'args': {'name': '{}_{:d}'.format(self.activation_name, layer_idx)},
            }
            all_layer_list.append(activation_function_dict)
            # Pooling layer
            if not self.poolstrides[layer_idx] == 1:
                pool_layer_dict = {
                    'layer_type': self.poolmethod,
                    'args': {
                        'strides': self.poolstrides[layer_idx],
                        'pool_size': self.pool_size[layer_idx],
                        'padding': self.pool_padding,
                        'name': 'pool_{:d}'.format(layer_idx), 
                        **extra_pooling_args,
                    }
                }
                all_layer_list.append(pool_layer_dict)
            # Batch normalization layer
            if self.batch_normalization:
                batch_norm_layer_dict = {
                    'layer_type': 'tf.layers.batch_normalization',
                    'args': {'name':'batch_norm_{:d}'.format(layer_idx)},
                }
                all_layer_list.append(batch_norm_layer_dict)

        # Collapse dimensions after the final convolutional layer
        all_layer_list.append({'layer_type': 'tf.layers.flatten', 'args': {'name': 'flatten_end_conv'}})

        # Intermediate fully connected layer
        if self.include_fc_intermediate:
            fc_layer_dict = {
                'layer_type': 'tf.layers.dense',
                'args': {
                    'units': self.num_fc_units, 
                    'activation': None,
                    'name':'fc_intermediate',
                },
            }
            all_layer_list.append(fc_layer_dict)
            if self.fc_activation:
                activation_function_dict = {
                    'layer_type': self.activation_type,
                    'args': {'name': '{}_fc_intermediate'.format(self.activation_name)},
                }
                all_layer_list.append(activation_function_dict)
            if self.fc_batch_norm:
                batch_norm_layer_dict = {
                    'layer_type': 'tf.layers.batch_normalization',
                    'args': {'name': 'batch_norm_fc_intermediate'},
                }
                all_layer_list.append(batch_norm_layer_dict)
        
        # Dropout layer
        if self.dropout:
            dropout_layer_dict = {
                'layer_type': 'tf.layers.dropout',
                'args': {'rate':self.rate, 'name': 'dropout'},
            }
            all_layer_list.append(dropout_layer_dict)
        # Top classification layer
        if self.include_classification_layer:
            class_layer_dict = {
                'layer_type':'fc_top_classification',
                'args': {'name': 'fc_top', 'activation': None},
            }
            all_layer_list.append(class_layer_dict)
        
        self.all_layer_list = all_layer_list


def uniformly_sample_conv_layers(range_num_conv_layers,
                                 possible_conv_layer_height,
                                 possible_conv_layer_length,
                                 possible_conv_layer_nums,
                                 possible_conv_strides_height,
                                 possible_conv_strides_length,
                                 possible_pooling_strides_height,
                                 possible_pooling_strides_length):
    '''
    Uniformly samples from the options within the provided inputs.
    
    Args
    ----
    range_num_conv_layers (list): possible depths of the network (number of conv layers)
    possible_conv_layer_height (list): for each convolutional layer, the height of the kernels
    possible_conv_layer_length (list): for each convolutional layer, the width of the kernels
    possible_conv_layer_nums (list): for each convolutional layer, the number of kernels to include
    possible_conv_strides_height (list): for each conv layer, the possible stides (height)
    possible_conv_strides_length (list): for each conv layer, the possible strides (width)
    possible_pooling_strides_height (list): for each pooling layer, the amount of pooling (height)
    possible_pooling_strides_length (list): for each pooling layer, the amount of pooling (width)
    
    Returns
    -------
    num_conv_kernels (list):
    convstrides (list):
    poolstrides (list):
    conv_kernels_sizes (list):
    '''
    num_conv_kernels = []
    convstrides = []
    poolstrides = []
    conv_kernels_sizes = []
    num_conv_layers = np.random.randint(range_num_conv_layers[0], range_num_conv_layers[1]+1)
    for layer_idx in np.arange(num_conv_layers):
        # Sample number of convolutional kernels in each layer
        rand_choose_num_conv = np.random.randint(0, len(possible_conv_layer_nums[layer_idx]))
        num_conv_kernels.append(possible_conv_layer_nums[layer_idx][rand_choose_num_conv])
        # Sample the stride for each layer
        rand_choose_conv_stride_height = np.random.randint(0, len(possible_conv_strides_height[layer_idx]))
        rand_choose_conv_stride_length = np.random.randint(0, len(possible_conv_strides_length[layer_idx]))
        convstrides.append([possible_conv_strides_height[layer_idx][rand_choose_conv_stride_height],
                            possible_conv_strides_length[layer_idx][rand_choose_conv_stride_length]])
        # Sample the pooling for each layer
        rand_choose_poolstrides_height = np.random.randint(0, len(possible_pooling_strides_height[layer_idx]))
        rand_choose_poolstrides_length = np.random.randint(0, len(possible_pooling_strides_length[layer_idx]))
        poolstrides.append([possible_pooling_strides_height[layer_idx][rand_choose_poolstrides_height],
                            possible_pooling_strides_length[layer_idx][rand_choose_poolstrides_length]])
        # Sample the convolutional filter height and length for each layer
        rand_choose_conv_height = np.random.randint(0, len(possible_conv_layer_height[layer_idx]))
        rand_choose_conv_length = np.random.randint(0, len(possible_conv_layer_length[layer_idx]))
        conv_kernels_sizes.append([possible_conv_layer_height[layer_idx][rand_choose_conv_height],
                                   possible_conv_layer_length[layer_idx][rand_choose_conv_length]])
    return num_conv_kernels, convstrides, poolstrides, conv_kernels_sizes


def generate_network_architecture(save_net_name):
    '''
    Generates a new network architecture using the prior specified in the RandomCNN class
    
    Args
    ----
    save_net_name (string) : the location of the generated json file
    
    Returns
    -------
    new_net (class) : an instance of the network class that has a json saved to save_net_name
    net_json_path (string) : location where the network json path was stored
    '''
    new_net = RandomCNN()
    print('Making new network architecture')
    net_json_path = save_net_name + '.json'
    with open(net_json_path, 'w') as fp:
        json.dump(new_net.all_layer_list, fp)
    for layer in new_net.all_layer_list:
        print(layer)
    return new_net, net_json_path


def main():
    generate_network_architecture(sys.argv[1])


if __name__ == "__main__":
    main() 
