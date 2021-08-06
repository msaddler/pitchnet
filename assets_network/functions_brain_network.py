import os
import sys
import json
import functools
import pickle
import numpy as np
import tensorflow as tf

from functions_parameter_handling import are_identical_dicts

sys.path.append(os.path.abspath(os.path.join(sys.path[0], '../msutil')))
import util_stimuli


def make_brain_net(input_tensor, n_classes_dict, brain_net_architecture, trainable=True, batchnorm_flag=True, dropout_flag=True, save_arch_path=None, save_pckl_path=None, only_include_layers=None):
    """
    Constructs a network specified by the parameters within brain_net_architecture
    Contains everything up to the final softmax layer (as this can change depending on the task)

    Args
    ----
    input_tensor (tensorflow tensor) : the input to the graph. 
    n_classes_dict (dictionary) : keys are the task paths, values are the number of classes for the task
    brain_net_architecture (dict) : a dictionary containing the architecture to build. Can be constructed with 'random_architecture_generator'
    trainable (Boolean, or tf placeholder) : whether to set the variables in the graph to trainable
    batchnorm_flag (Boolean, or tf placeholder) : whether to run batch norm in training (True) or testing (False) mode. 
    dropout_flag (Boolean, or tf placholder) : whether to turn on dropout 
    save_arch_path (string) : path where a .json should be saved containing the graph architecture
    save_pckl_path (string) : location to save a pckled version with partial functions to rebuild the graph
    only_include_layers (set) : if not None, stops building the graph when all of the keys included in this set are constructed

    Returns
    -------
    output_tensor (tensorflow tensor) : last tensor in the graph
    nets (dict) : dictionary containing the full brain graph
    """
    layer_list = []
    nets = {}
    # Enumerate through layers, keep track of numbers within brain dict to not overwrite anything. 
    for layer_idx, layer in enumerate(brain_net_architecture):
        assert (layer['args']['name'] not in nets.keys()), 'Brain architecture should not have multiple layers with the same name.'
        if layer['layer_type'] == 'tf.layers.conv2d':
            layer_list.append(functools.partial(conv2d_valid_width_wrapper, **layer['args']))
            nets[layer['args']['name']] = layer_list[-1](input_tensor, trainable=trainable)

        elif layer['layer_type'] == 'tf.nn.relu':
            layer_list.append(functools.partial(tf.nn.relu, **layer['args']))
            nets[layer['args']['name']] = layer_list[-1](input_tensor)

        elif layer['layer_type'] == 'tf.nn.relu6':
            layer_list.append(functools.partial(tf.nn.relu6, **layer['args']))
            nets[layer['args']['name']] = layer_list[-1](input_tensor)
        
        elif layer['layer_type'] == 'tf.nn.leaky_relu': 
            layer_list.append(functools.partial(tf.nn.leaky_relu, **layer['args']))
            nets[layer['args']['name']] = layer_list[-1](input_tensor)      

        elif layer['layer_type'] == 'tf.layers.batch_normalization':
            layer_list.append(functools.partial(tf.layers.batch_normalization, **layer['args'], fused=False)) # turn off fused batch noormalization due to bug with using placeholders 
            nets[layer['args']['name']] = layer_list[-1](input_tensor, training=batchnorm_flag, trainable=trainable)

        elif layer['layer_type'] == 'hpool':
            layer_list.append(functools.partial(hanning_pooling, **layer['args'])) 
            nets[layer['args']['name']] = layer_list[-1](input_tensor)

        elif layer['layer_type'] == 'tf.layers.max_pooling2d':
            layer_list.append(functools.partial(tf.layers.max_pooling2d, **layer['args']))
            nets[layer['args']['name']] = layer_list[-1](input_tensor)

        elif layer['layer_type'] == 'tf.layers.average_pooling2d':
            layer_list.append(functools.partial(tf.layers.average_pooling2d, **layer['args']))
            nets[layer['args']['name']] = layer_list[-1](input_tensor)
 
        elif layer['layer_type'] == 'tf.nn.avg_pool':
            layer_list.append(functools.partial(tf.nn.avg_pool, **layer['args']))
            nets[layer['args']['name']] = layer_list[-1](input_tensor)

        elif layer['layer_type'] == 'tf.layers.flatten':
            layer_list.append(functools.partial(tf.layers.flatten, **layer['args']))
            nets[layer['args']['name']] = layer_list[-1](input_tensor)

        elif layer['layer_type'] == 'tf.layers.dense':
            layer_list.append(functools.partial(tf.layers.dense, **layer['args']))
            nets[layer['args']['name']] = layer_list[-1](input_tensor, trainable=trainable)

        elif layer['layer_type'] == 'tf.layers.dropout':
            layer_list.append(functools.partial(tf.layers.dropout, **layer['args']))
            nets[layer['args']['name']] = layer_list[-1](input_tensor, training=dropout_flag)

        elif layer['layer_type'] == 'fc_top_classification':
            layer_list.append(functools.partial(fc_top_classification, n_classes_dict=n_classes_dict, **layer['args']))
            nets[layer['args']['name']] = layer_list[-1](input_tensor, trainable=trainable)            

        elif layer['layer_type'] == 'multi_fc_top_classification':
            layer_list.append(functools.partial(multi_fc_top_classification, n_classes_dict=n_classes_dict, **layer['args'])) 
            nets[layer['args']['name']] = layer_list[-1](input_tensor, trainable=trainable) 
 
        elif layer['layer_type'] == 'tf.pad':
            layer_list.append(functools.partial(tf.pad, **layer['args']))
            nets[layer['args']['name']] = layer_list[-1](input_tensor)

        elif layer['layer_type'] == 'tf.slice':
            part = functools.partial(tf.slice, **layer['args'])
            layer_list.append(part)
            nets[layer['args']['name']] = part(input_tensor)

        elif layer['layer_type'] == 'tf.transpose':
            part = functools.partial(tf.transpose, **layer['args'])
            layer_list.append(part)
            nets[layer['args']['name']] = part(input_tensor)

        elif layer['layer_type'] == 'tfnnresample':
            def tfnnresample_wrapper(tensor_input,
                                     sr_input,
                                     sr_output,
                                     kwargs_nnresample_poly_filter={},
                                     **kwargs):
                """Wrapper designed to ignore layer['args']['name']"""
                return util_stimuli.tfnnresample(
                    tensor_input,
                    sr_input,
                    sr_output,
                    kwargs_nnresample_poly_filter=kwargs_nnresample_poly_filter)
            part = functools.partial(tfnnresample_wrapper, **layer['args'])
            layer_list.append(part)
            nets[layer['args']['name']] = part(input_tensor)

        else: 
            print('Layer type %s is not supported, check config file'%layer['layer_type'])
        # update the input tensor
        input_tensor = nets[layer['args']['name']]
        # If we are only building part of the graph, then check to see if we should stop building
        if only_include_layers is not None:
            assert type(only_include_layers)==set, 'If only_include_layers is not None, it must be a set of layers to check for inclusion'
            if layer['args']['name'] in only_include_layers:
                only_include_layers.remove(layer['args']['name'])
            # If the set is empty, then break from the loop and don't build any more of the graph
            if only_include_layers == set():
                break

    if save_arch_path is not None: 
        if os.path.isfile(save_arch_path): # if identical, then don't need to overwrite the file
            with open(save_arch_path, 'r') as load_f: loaded_arch = json.load(load_f)
            assert are_identical_dicts(loaded_arch, brain_net_architecture), "The saved brain architecture in %s does not match the one you are trying to build"%save_arch_path
        else: 
            with open(save_arch_path, 'w') as arch_f:
                json.dump(brain_net_architecture, arch_f)

    if save_pckl_path is not None: 
        to_pickle = {'instructions':'To rebuild the graph, loop through each element of layer_list. Only input arguments to the functions should be "trainable" and "training", which are in relation to variable creation, batch norm, and dropout.', 'layer_list':layer_list, 'graph_architecture':brain_net_architecture}
        # In the output file, one needs to loop through each element of the list in order to build the graph. 
        with open(save_pckl_path, 'wb') as pickle_file:
            pickle.dump(to_pickle, pickle_file)
    output_tensor = input_tensor
    return output_tensor, nets


def fc_top_classification(input_tensor, n_classes_dict, **kwargs):
    """
    Builds an fc layer at the top of the network for classification, parses n_classes_dict.

    Args
    ----
    input_tensor (tensorflow tensor) : the input layer for each of the added fc layers
    n_classes_dict (dict) : contains the number of classes (number of FC units) for each of the tasks
    kwargs : keyword arguments to pass into tf.layers.dense

    Outputs
    -------
    output_tensor (tensorflow tensor) : an fc layer with the number of classes
 
    """

    assert len(list(n_classes_dict.keys())) == 1, "Multiple tasks specified but only one FC layer can be constructed with 'fc_top_classification', please check network configuration."
    (task_name, task_classes), = n_classes_dict.items()
    output_tensor = tf.layers.dense(input_tensor, units=task_classes, **kwargs)
    return output_tensor


def multi_fc_top_classification(input_tensor, n_classes_dict, name, **kwargs):
    """
    Builds multiple FC layers (and appends integer names) for each of the tasks specified in n_classes_dict.  
    
    Args 
    ----
    input_tensor (tensorflow tensor) : the input layer for each of the added fc layers
    n_classes_dict (dict) : contains the number of classes (number of FC units) for each of the tasks 
    name (string) : name of the fc_layer, function appends integers to name for each task

    Outputs
    -------
    output_layer_dict (dictionary) : dictionary containing each of the output fc layers

    """
    output_layer_dict = {}
    all_keys_tasks = list(n_classes_dict.keys())
    all_keys_tasks.sort() # so that when we reload things are in the same order
    for num_classes_task_idx, num_classes_task_name in enumerate(all_keys_tasks):
        task_name = '%s_%s'%(name, num_classes_task_idx)
        output_layer_dict[num_classes_task_name] = tf.layers.dense(input_tensor, units=n_classes_dict[num_classes_task_name], name=task_name, **kwargs)

    return output_layer_dict


def hanning_pooling(input_layer, strides=2, pool_size=8, padding='SAME', name=None, sqrt_window=False, normalize=False):
    """
    Add a layer using a hanning kernel for pooling

    Parameters
    ----------
    input_layer : tensorflow tensor
        layer to add hanning pooling to
    strides : int
        proportion downsampling
    top_node : string
        specify the node after which the spectemp filters will be added and used as input for the FFT.
    sqrt_window : boolean
        if true, takes the sqrt of the window (old version), normal window generation has sqrt_window=False
    normalize : boolean
        if true, divide the filter by the sum of its values, so that the smoothed signal is the same amplitude as the original.
    name : False or string
        name for the layer. If false appends "_hpool" to the top_node name

    Returns
    -------
    output_layer : tensorflow tensor
        input_layer with hanning pooling applied
    """
    n_channels = input_layer.get_shape().as_list()[3]
    hanning_window_tensor = make_hanning_kernel_tensor_no_depthwise(n_channels, strides=strides, pool_size=pool_size, sqrt_window=sqrt_window, normalize=normalize, name='%s_hpool_kernel'%name)
    if type(strides)!=list and type(strides)==int:
        strides = [strides, strides] # using square filters
    output_layer = conv2d_for_hpool_valid_width_wrapper(input_layer, filters=hanning_window_tensor, strides=[1, strides[0], strides[1], 1], padding=padding, name=name)
    return output_layer


def make_hanning_kernel_tensor_no_depthwise(n_channels, strides=2, pool_size=8, sqrt_window=False, normalize=False, name=None):
    """
    Make a tensor containing the symmetric 2d hanning kernel to use for the pooling filters
    For strides=2, using pool_size=8 gives a reduction of -24.131545969216841 at 0.25 cycles
    For strides=3, using pool_size=12 gives a reduction of -28.607805482176282 at 1/6 cycles

    This version uses the normal conv2d operation and fills most of the smoothing tensor with zeros. Depthwise convolution
    does not have a second order gradient, and cannot be used with some functions.

    Parameters
    ----------
    n_channels : int
        number of channels to copy the kernel into
    strides : int
        proportion downsampling
    pool_size : int
        how large of a window to use
    sqrt_window : boolean
        if true, takes the sqrt of the window (old version), normal window generation has sqrt_window=False
    normalize : boolean
        if true, divide the filter by the sum of its values, so that the smoothed signal is the same amplitude as the original.
    name : False or string
        name for the layer. If false appends "_hpool" to the top_node name


    Returns
    -------
    hanning_tensor : tensorflow tensor
        tensorflow tensor containing the hanning kernel with size [1 pool_size pool_size n_channels]

    """
    hanning_kernel = make_hanning_kernel(strides=strides,pool_size=pool_size,sqrt_window=sqrt_window, normalize=normalize).astype(np.float32)
    hanning_kernel = np.expand_dims(np.expand_dims(hanning_kernel,0),0) * np.expand_dims(np.expand_dims(np.eye(n_channels),3),3) # [width, width, n_channels, n_channels]
    hanning_tensor = tf.constant(hanning_kernel, dtype=tf.float32, name=name)
    hanning_tensor = tf.transpose(hanning_tensor, [2,3,0,1])
    return hanning_tensor


def make_hanning_kernel(strides=2, pool_size=8, sqrt_window=False, normalize=False):
    """
    Make the symmetric 2d hanning kernel to use for the pooling filters
    For strides=2, using pool_size=8 gives a reduction of -24.131545969216841 at 0.25 cycles
    For strides=3, using pool_size=12 gives a reduction of -28.607805482176282 at 1/6 cycles

    Parameters
    ----------
    strides : int
        proportion downsampling
    pool_size : int
        how large of a window to use
    sqrt_window : boolean
        if true, takes the sqrt of the window (old version), normal window generation has sqrt_window=False
    normalize : boolean
        if true, divide the filter by the sum of its values, so that the smoothed signal is the same amplitude as the original.

    Returns
    -------
    two_dimensional_kernel : numpy array
        hanning kernel in 2d to use as a kernel for filtering

    """

    if type(strides)!=list and type(strides)==int:
        strides = [strides, strides] # using square filters
 
    if type(pool_size)!=list and type(pool_size)==int: 
        if pool_size > 1:
            window = 0.5 * (1 - np.cos(2.0 * np.pi * (np.arange(pool_size)) / (pool_size - 1)))
            if sqrt_window: 
                two_dimensional_kernel = np.sqrt(np.outer(window, window))
            else: 
                two_dimensional_kernel = np.outer(window, window)
        else: 
            window = np.ones((1,1))
            two_dimensional_kernel = window # [1x1 kernel]
    elif type(pool_size)==list:
        if pool_size[0] > 1:
            window_h = np.expand_dims(0.5 * (1 - np.cos(2.0 * np.pi * (np.arange(pool_size[0])) / (pool_size[0] - 1))),0)
        else:
            window_h = np.ones((1,1))
        if pool_size[1] > 1:
            window_w = np.expand_dims(0.5 * (1 - np.cos(2.0 * np.pi * (np.arange(pool_size[1])) / (pool_size[1] - 1))),1)
        else:
            window_w = np.ones((1,1))
 
        if sqrt_window:
            two_dimensional_kernel = np.sqrt(np.outer(window_h, window_w))
        else:  
            two_dimensional_kernel = np.outer(window_h, window_w)

    if normalize:
        two_dimensional_kernel = two_dimensional_kernel/(sum(two_dimensional_kernel.ravel()))        
    
    return two_dimensional_kernel


def conv2d_valid_width_wrapper(inputs,kernel_size,strides,padding,**kwargs):
    """
    Wraps tf.layers.conv2d to allow valid convolution across signal width and
    'same' convolution across signal height when padding is set to "valid_time"
    
  Arguments:
    inputs (TF Tensor): Tensor input.
    kernel_size (int or tuple/list): An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides (int or tuple/list) : An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding (string): One of `"valid"`, `"same"`, or `"valid_time"` (case-insensitive).
    kwargs (dictionary): Specifies all other arguments required by
    tf.layers.conv2d. Passes these directly to function without modification.
        See Tensorflow documentation for further details.

  Returns:
      (TF Tensor): Output of tf.layers.conv2d.
    """

    #Collects relvant parameters    
    size=inputs.get_shape()
    filter_height = kernel_size[0]
    in_height = size[1]

    #Calculates according to SAME padding formula
    if (in_height % strides[0] == 0):
        pad_along_height = max(filter_height - strides[0], 0)
    else:
        pad_along_height = max(filter_height - (in_height % strides[0]), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top

    #Pads signal if VALID_TIME is selected and padding is necessary
    #Otherwise, pass inputs through and allow specified convolutioon
    if pad_along_height == 0 or padding.upper() != 'VALID_TIME':
        padding = 'VALID' if padding.upper() == 'VALID_TIME' else padding
        output_tensor = tf.layers.conv2d(inputs,kernel_size=kernel_size,
                                         strides=strides,padding=padding,
                                         **kwargs)
    else:
        #Pads input tensor and moves conv2d to valid padding
        paddings = tf.constant([[0,0],[pad_top, pad_bottom], [0, 0],[0,0]])
        input_padded = tf.pad(inputs,paddings)
        output_tensor=tf.layers.conv2d(input_padded,kernel_size=kernel_size,
                                       strides=strides, padding="VALID",
                                       **kwargs)
    return output_tensor


def conv2d_for_hpool_valid_width_wrapper(inputs,filters,strides,padding,**kwargs):
    """
    Wraps tf.layers.conv2d to allow valid convolution across signal width and
    'same' convolution across signal height when padding is set to "valid_time"
    
  Arguments:
    inputs (TF Tensor): Tensor input.
    filters (TF Tensor):  Must have the same type as input.
      A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
    strides (int or tuple/list) : An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding (string): One of `"valid"`, `"same"`, or `"valid_time"` (case-insensitive).
    kwargs (dictionary): Specifies all other arguments required by
    tf.layers.conv2d. Passes these directly to function without modification.
        See Tensorflow documentation for further details.

  Returns:
      (TF Tensor): Output of tf.layers.conv2d.
    """

    #Collects relvant parameters    
    size=inputs.get_shape()
    kernel_size = filters.get_shape()
    filter_height = int(kernel_size[0])
    in_height = int(size[1])

    #Calculates according to SAME padding formula
    if (in_height % strides[0] == 0):
        pad_along_height = max(filter_height - strides[0], 0)
    else:
        pad_along_height = max(filter_height - (in_height % strides[0]), 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top

    #Pads signal if VALID_TIME is selected and padding is necessary
    #Otherwise, pass inputs through and allow specified convolutioon
    if pad_along_height == 0 or padding.upper() != 'VALID_TIME':
        padding = 'VALID' if padding.upper() == 'VALID_TIME' else padding
        output_tensor = tf.nn.conv2d(inputs,filter=filters,
                                         strides=strides,padding=padding,
                                         **kwargs)
    else:
        #Pads input tensor and moves conv2d to valid padding
        paddings = tf.constant([[0,0],[pad_top, pad_bottom], [0, 0],[0,0]])
        input_padded = tf.pad(inputs,paddings)
        output_tensor=tf.nn.conv2d(input_padded,filter=filters,
                                       strides=strides, padding="VALID",
                                       **kwargs)
    return output_tensor
