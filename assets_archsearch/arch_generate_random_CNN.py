import numpy as np


def get_random_cnn_architecture(kwargs_sample_repeating_cnn_elements={},
                                dilation_rate=[1,1],
                                activation_type='tf.nn.relu',
                                activation_name='relu',
                                pool_method='hpool',
                                include_batch_normalization=True,
                                range_intermediate_fc_layer=[False, True],
                                range_intermediate_fc_layer_units=[128, 256, 512, 1024],
                                include_fc_batch_normalization=True,
                                include_fc_activation=True,
                                include_dropout=True,
                                dropout_rate=0.5,
                                include_classification_layer=True):
    '''
    '''
    ### ------ Randomly sample CNN architecture ------ ###
    repeating_cnn_elements = sample_repeating_cnn_elements(**kwargs_sample_repeating_cnn_elements)
    intermediate_fc_layer = {}
    if np.random.choice(range_intermediate_fc_layer):
        intermediate_fc_layer['units'] = np.random.choice(range_intermediate_fc_layer_units)
    
    ### ------ Build the sampled CNN architecture ------ ###
    conv_layer_count = repeating_cnn_elements['conv_layer_count']
    layer_list = []
    for layer_index in range(conv_layer_count):
        # Repeating elements: convolution operation
        conv_layer_dict = {
            'layer_type': 'tf.layers.conv2d',
            'args': {
                'filters': repeating_cnn_elements['conv_kernel_depths'][layer_index],
                'kernel_size': repeating_cnn_elements['conv_kernel_shapes'][layer_index],
                'padding': repeating_cnn_elements['conv_padding'],
                'strides': repeating_cnn_elements['conv_strides'][layer_index],
                'dilation_rate': dilation_rate,
                'activation': None,
                'name': 'conv_{:d}'.format(layer_index),
            },
        }
        layer_list.append(conv_layer_dict)
        # Repeating elements: activation function
        activation_function_dict = {
            'layer_type': activation_type,
            'args': {
                'name': '{}_{:d}'.format(activation_name, layer_index)
            },
        }
        layer_list.append(activation_function_dict)
        # Repeating elements: pooling operation
        if pool_method == 'hpool':
            pool_size = repeating_cnn_elements['pool_kernel_shapes'][layer_index]
            extra_pooling_args = {'normalize': True, 'sqrt_window': False}
        elif pool_method == 'tf.layers.max_pooling2d':
            pool_size = repeating_cnn_elements['pool_strides'][layer_index]
            extra_pooling_args = {}
        else:
            raise ValueError("pool_method={} is not supported".format(pool_method))
        pool_layer_dict = {
            'layer_type': pool_method,
            'args': {
                'strides': repeating_cnn_elements['pool_strides'][layer_index],
                'pool_size': pool_size,
                'padding': repeating_cnn_elements['pool_padding'],
                'name': 'pool_{:d}'.format(layer_index), 
                **extra_pooling_args,
            }
        }
        layer_list.append(pool_layer_dict)
        # Repeating elements: batch normalization operation
        if include_batch_normalization:
            batch_norm_layer_dict = {
                'layer_type': 'tf.layers.batch_normalization',
                'args': {
                    'name':'batch_norm_{:d}'.format(layer_index)
                },
            }
            layer_list.append(batch_norm_layer_dict)
    
    # Flatten representation following final repeating element
    layer_list.append({'layer_type': 'tf.layers.flatten', 'args': {'name': 'flatten_end_conv'}})
    
    # Intermediate fully-connected layer (dense, activation, normalization)
    if intermediate_fc_layer:
        fc_layer_dict = {
            'layer_type': 'tf.layers.dense',
            'args': {
                'units': intermediate_fc_layer['units'], 
                'activation': None,
                'name':'fc_intermediate',
            },
        }
        layer_list.append(fc_layer_dict)
        if include_fc_activation:
            activation_function_dict = {
                'layer_type': activation_type,
                'args': {
                    'name': '{}_fc_intermediate'.format(activation_name)
                },
            }
            layer_list.append(activation_function_dict)
        if include_fc_batch_normalization:
            batch_norm_layer_dict = {
                'layer_type': 'tf.layers.batch_normalization',
                'args': {
                    'name': 'batch_norm_fc_intermediate'
                },
            }
            layer_list.append(batch_norm_layer_dict)
    
    # Dropout layer
    if include_dropout:
        dropout_layer_dict = {
            'layer_type': 'tf.layers.dropout',
            'args': {
                'rate':dropout_rate,
                'name': 'dropout'
            },
        }
        layer_list.append(dropout_layer_dict)
    
    # Final classification layer
    if include_classification_layer:
        class_layer_dict = {
            'layer_type':'fc_top_classification',
            'args': {
                'name': 'fc_top', 'activation': None
            },
        }
        layer_list.append(class_layer_dict)
    
    return layer_list, repeating_cnn_elements


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
        pool_strides.append([pool_stride_dim1, pool_stride_dim2])
        pool_kernel_shape = [pool_stride_dim1, pool_stride_dim2]
        if pool_kernel_shape[0] > 1:
            pool_kernel_shape[0] = pool_kernel_shape[0] * pool_kernel_size_dim1
        if pool_kernel_shape[1] > 1:
            pool_kernel_shape[1] = pool_kernel_shape[1] * pool_kernel_size_dim2
        pool_kernel_shapes.append(pool_kernel_shape)
    
    # Return description of repeating CNN elements in a single dictionary
    repeating_cnn_elements = {
        'conv_layer_count': conv_layer_count,
        'conv_kernel_depths': conv_kernel_depths,
        'conv_kernel_shapes': conv_kernel_shapes,
        'conv_strides': conv_strides,
        'conv_padding': conv_padding,
        'pool_kernel_shapes': pool_kernel_shapes,
        'pool_strides': pool_strides,
        'pool_padding': pool_padding,
        'max_kernel_area': max_kernel_area,
    }
    return repeating_cnn_elements
