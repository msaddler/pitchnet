import sys
import os
import numpy as np
import json


def get_random_cnn_architecture():
    pass


def sample_repeating_cnn_elements(input_shape=[None, 100, 1000, 1],
                                  range_conv_layer_count=[1, 8],
                                  max_kernel_area=500,
                                  range_conv_kernel_dim1=[1e-3, 5e-1],
                                  range_conv_kernel_dim2=[1e-1, 5e-1],
                                  range_conv_stride_dim1=[1],
                                  range_conv_stride_dim2=[1],
                                  range_conv_depth=[8, 1024],
                                  range_conv_depth_step=[0.5, 1.0, 1.0, 2.0, 2.0, 2.0],
                                  range_conv_depth_initial=[8, 16, 32, 64],
                                  range_pool_stride_dim1=[1, 3],
                                  range_pool_stride_dim2=[1, 8],
                                  pool_kernel_size_dim1=4,
                                  pool_kernel_size_dim2=4,
                                  conv_padding='VALID',
                                  pool_padding='SAME'):
    '''
    '''
    # Initialize output lists
    conv_kernel_depths = []
    conv_kernel_shapes = []
    conv_strides = []
    pool_kernel_shapes = []
    pool_strides = []
    
    # Sample number of repeated layers and iterate over them
    current_dims = np.array(input_shape)
    conv_layer_count = np.random.randint(low=range_conv_layer_count[0],
                                         high=range_conv_layer_count[1]+1,
                                         size=None, dtype=int)
    for layer_index in range(conv_layer_count):
        # Sample conv layer depth
        if layer_index == 0:
            current_dims[3] = np.random.choice(range_conv_depth_initial)
        else:
            depth_step = np.random.choice(range_conv_depth_step)
            current_dims[3] = int(depth_step * current_dims[3])
            if current_dims[3] < range_conv_depth[0]:
                current_dims[3] = range_conv_depth[0]
            if current_dims[3] > range_conv_depth[1]:
                current_dims[3] = range_conv_depth[1]
        conv_kernel_depths.append(current_dims[3])
        
        # Sample conv kernel shapes and strides
        acceptable_kernel_shape = False
        while not acceptable_kernel_shape:
            fract_dim1 = np.random.uniform(low=range_conv_kernel_dim1[0],
                                           high=range_conv_kernel_dim1[1])
            fract_dim2 = np.random.uniform(low=range_conv_kernel_dim2[0],
                                           high=range_conv_kernel_dim2[1])
            kernel_dim1 = np.ceil(current_dims[1] * fract_dim1).astype(int)
            kernel_dim2 = np.ceil(current_dims[2] * fract_dim2).astype(int)
            if kernel_dim1 * kernel_dim2 <= max_kernel_area:
                acceptable_kernel_shape = True
        conv_stride_dim1 = np.random.choice(range_conv_stride_dim1)
        conv_stride_dim2 = np.random.choice(range_conv_stride_dim2)
        assert conv_stride_dim1 == 1, "strided convolution is not supported"
        assert conv_stride_dim2 == 1, "strided convolution is not supported"
        if conv_padding == 'VALID':
            current_dims[1] = np.ceil((current_dims[1] - kernel_dim1 + 1) / conv_stride_dim1).astype(int)
            current_dims[2] = np.ceil((current_dims[2] - kernel_dim2 + 1) / conv_stride_dim2).astype(int)
        else:
            raise ValueError("conv_padding={} is not supported".format(conv_padding))
        conv_kernel_shapes.append([kernel_dim1, kernel_dim2])
        conv_strides.append([conv_stride_dim1, conv_stride_dim2])
        
        # Sample pool kernel shapes and strides
        dim1_range = np.array(range_pool_stride_dim1)
        dim1_range[1] = min(dim1_range[1], current_dims[1]/pool_kernel_size_dim1)
        dim1_range[1] = max(1, dim1_range[1])
        dim2_range = np.array(range_pool_stride_dim2)
        dim2_range[1] = min(dim2_range[1], current_dims[2]/pool_kernel_size_dim2)
        dim2_range[1] = max(1, dim2_range[1])
        pool_stride_dim1 = np.random.randint(low=dim1_range[0], high=dim1_range[1]+1, dtype=int)
        pool_stride_dim2 = np.random.randint(low=dim2_range[0], high=dim2_range[1]+1, dtype=int)
        if pool_padding == 'SAME':
            current_dims[1] = np.ceil(current_dims[1] / pool_stride_dim1).astype(int)
            current_dims[2] = np.ceil(current_dims[2] / pool_stride_dim2).astype(int)
        else:
            raise ValueError("pool_padding={} is not supported".format(pool_padding))
        print((kernel_dim1, kernel_dim2), (pool_stride_dim1, pool_stride_dim2), current_dims)
    
    return conv_kernel_depths, conv_kernel_shapes, conv_strides, pool_kernel_shapes, pool_strides



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
        
        # Discrete prior over number of convolutional layers (sampled using np.random.choice())
        self.possible_num_conv_layers = [
            1,
            2,2,
            3,3,3,
            4,4,4,4,
            5,5,5,5,5,
            6,6,6,6,6,6,
            7,7,7,7,7,7,7,
            8,8,8,8,8,8,8,8,
        ]
        
        # Discrete priors over convolutional kernel heights (freq) and lengths (time) for each layer  
        self.possible_conv_layer_height = [
            [1,2,3,4,5],
            [1,2,3,4,5],
            [1,2,3,4,5],
            [1,2,3,4,5],
            [1,2,3,4,5],
            [1,2,3,4,5],
            [1,2,3,4,5],
            [1,2,3,4,5],
        ]
        self.possible_conv_layer_length = [
            [16,24,32,48,64,96,128],
            [8,12,16,24,32,48,64],
            [4,8,12,16,24,32],
            [2,4,8,12,16],
            [1,2,4,8],
            [1,2,3,4],
            [1,2,3,4],
            [1,2,3,4],
        ]
        # Discrete prior over number of convolutional kernels for each layer
        self.possible_conv_layer_nums = [
            [16,32,64,128],
            [16,32,64,128,256],
            [32,64,128,256],
            [32,64,128,256,512],
            [64,128,256,512],
            [64,128,256,512],
            [128,256,512],
            [128,256,512,1024],
        ]
        
        # Discrete priors over convolutional strides for each layer
        self.possible_conv_strides_height = [[1]] * np.max(self.possible_num_conv_layers) # NO STRIDED CONVOLUTION
        self.possible_conv_strides_length = [[1]] * np.max(self.possible_num_conv_layers) # NO STRIDED CONVOLUTION
        
        # Discrete priors over pooling sizes after each conv layer ("1" pooling = no pooling)
        self.possible_pooling_strides_height = [
            [1,2,3],
            [1,2,3],
            [1,2,3],
            [1,2],
            [1,2],
            [1,1,2],
            [1,1,2],
            [1,1,2],
        ]
        self.possible_pooling_strides_length = [
            [2,4,8],
            [1,2,4],
            [1,2,4],
            [1,2],
            [1,2],
            [1,1,2],
            [1,1,1,2],
            [1,1,1,1,2],
        ]
        
        # Discrete prior over existence and size of an intermediate fully-connected layer
        self.possible_layer_fc=[False,True]
        self.possible_num_fc_units=[128,256,512,1024]
        
        ###### SAMPLE RANDOM ARCHITECTURE FROM THE PRIORS SET ABOVE ######
        
        # Choose the parameters for the conv layers
        self.num_conv_kernels, self.convstrides, self.poolstrides, self.conv_kernels_sizes = uniformly_sample_conv_layers(
            self.possible_num_conv_layers,
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


def uniformly_sample_conv_layers(possible_num_conv_layers,
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
    possible_num_conv_layers (list): possible depths of the network (number of conv layers)
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
    num_conv_layers = np.random.choice(possible_num_conv_layers)
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
