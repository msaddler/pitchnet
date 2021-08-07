import tensorflow as tf
import numpy as np


def make_task_evaluation_metrics(batch_out_dict, batch_labels_dict, batch_loss_dict,
                                 batch_audio_dict={}, TASK_LOSS_PARAMS={}):
    """
    Builds the measures that will be used during the evaluation pipeline

    Args:
        batch_out_dict (dict of tensor) : logits output by the graph
        batch_labels_dict (dict of tensor) : labels for the input data
        batch_loss_dict (dict of losses) : contains the loss for each task
        batch_audio_dict (dict of tensor) : if nonempty, audio tensors will be included in all_evaluation_measures_dict
            (possible keys are `input_audio` and `wavenet_audio`)
        TASK_LOSS_PARAMS (dict): dictionary of task-specific loss function parameters
    Returns:
        all_evaluation_measures_dict (dict) : keys are names of evaluation measures, elements are tensors to run the metric
    """
    all_evaluation_measures = [] 
    all_evaluation_measures_names = []
    all_evaluation_measures_dict = {}
    
    if TASK_LOSS_PARAMS is None: TASK_LOSS_PARAMS = {}
    # Supported activation functions (copied from `fga.build_task_loss_graph()`)
    activation_functions = {
        None: None,
        'identity': tf.identity,
        'linear': tf.identity,
        'relu': tf.nn.relu,
        'sigmoid': tf.math.sigmoid,
        'softmax': tf.nn.softmax,
    }
    
    # TODO: the update ops should be added to a collection so that we aren't saving them.
    # TODO: need way to decide which metrics to use that does not rely on tensor shapes (perhaps use TASK_LOSS_PARAMS)
    
    for task_key in batch_labels_dict.keys():
        task_labels = batch_labels_dict[task_key]
        task_logits = batch_out_dict[task_key]
        task_loss_params = TASK_LOSS_PARAMS.get(task_key, {})
        
        if (len(task_labels.shape) <= 1) and (task_logits.shape[1] == 1): # Single value regression task evaluation
            task_activation_fcn = activation_functions[task_loss_params.get('activation_type', None)]
            if task_activation_fcn is None: labels_pred = tf.squeeze(task_logits, axis=-1)
            else: labels_pred = tf.squeeze(task_activation_fcn(task_logits), axis=-1)
            labels_true = task_labels
            l1_loss, l1_loss_update_op = tf.metrics.mean(tf.math.abs(tf.math.subtract(labels_pred, labels_true)))
            l2_loss, l2_loss_update_op = tf.metrics.mean(tf.squared_difference(labels_pred, labels_true))
            all_evaluation_measures_dict[task_key + ':labels_true'] = labels_true
            all_evaluation_measures_dict[task_key + ':labels_pred'] = labels_pred
            all_evaluation_measures_dict[task_key + ':l1_loss'] = l1_loss
            all_evaluation_measures_dict[task_key + ':l1_loss_update_op'] = l1_loss_update_op
            all_evaluation_measures_dict[task_key + ':l2_loss'] = l2_loss
            all_evaluation_measures_dict[task_key + ':l2_loss_update_op'] = l2_loss_update_op
        
        elif len(task_labels.shape) <= 1: # One-hot task evaluation
            labels_true = tf.cast(task_labels, tf.int64)
            all_evaluation_measures_dict[task_key + ':labels_true'] = labels_true
            
            labels_pred = tf.argmax(task_logits, axis=1)
            all_evaluation_measures_dict[task_key + ':labels_pred'] = labels_pred
            
            accuracy, accuracy_update_op = tf.metrics.accuracy(labels_true, labels_pred)
            all_evaluation_measures_dict[task_key + ':accuracy'] = accuracy
            all_evaluation_measures_dict[task_key + ':accuracy_update_op'] = accuracy_update_op
            
            correct_pred = tf.equal(labels_pred, labels_true)
            all_evaluation_measures_dict[task_key + ':correct_pred'] = correct_pred
            
            probs_out = tf.nn.softmax(task_logits)
            all_evaluation_measures_dict[task_key + ':probs_out'] = probs_out
            
        else: # Multi-hot task evaluation
            probs_out = tf.sigmoid(task_logits)
            all_evaluation_measures_dict[task_key + ':probs_out'] = probs_out
            
            labels_true = tf.cast(task_labels, tf.float32)
            all_evaluation_measures_dict[task_key + ':labels_true'] = labels_true
            
            # TODO: We do not need to evaluate the AUC and the MAP every time... we just need to run the update. It would save time if they were not run. 
            auc, auc_update_op = tf.metrics.auc(labels_true, probs_out) 
            all_evaluation_measures_dict[task_key + ':auc'] = auc
            all_evaluation_measures_dict[task_key + ':auc_update_op'] = auc_update_op # AUC needs many samples... calculate over many batches.
            
            mean_average_precision, map_update = tf.metrics.average_precision_at_k(tf.cast(labels_true, tf.int64), probs_out, 20)
            all_evaluation_measures_dict[task_key + ':mAP@20'] = mean_average_precision
            all_evaluation_measures_dict[task_key + ':mAP@20_update_op'] = map_update #  MAP needs many samples... calculate over many batches.
    
    for loss_key in batch_loss_dict.keys():
        mean_loss, mean_loss_update_op = tf.metrics.mean(batch_loss_dict[loss_key])
        all_evaluation_measures_dict[loss_key + ':mean_loss_update_op'] = mean_loss_update_op
        all_evaluation_measures_dict[loss_key + ':mean_loss'] = mean_loss
        
    # Add "input_audio" and "wavenet_audio" (if applicable) keys to all_evaluation_measures_dict
    for audio_key in batch_audio_dict.keys():
        all_evaluation_measures_dict[audio_key] = batch_audio_dict[audio_key]
    
    return all_evaluation_measures_dict


class EarlyStopping:
    """Stop training when a monitored quantity has stopped improving.
    Adapted from Keras Library (AF 2019-04-15)
    # Arguments
        monitored_metrics_names (list) : list with quantities to be monitored.
        min_delta (float or dict) : minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement. If an integer is provided, that value will be used for
            all comparisons in `monitored_metrics_dict`. If a dictionary is
            provided, there must be a value present under every key name in
            `monitored_metrics_dict`.
        patience (int) : number of checks with no improvement
            after which training will be stopped.
        mode (str or dict): one of {auto, min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
            If a string is provided, that value will be used for
            all comparisons in `monitored_metrics_dict`. If a dictionary is
            provided, there must be a value present under every key name in
            `monitored_metrics_dict`.
        baseline (dictionary or None): Baseline values for the monitored quantity to reach.
            Training will stop if the model doesn't show improvement
            over the baseline. Will be ignored if set to None.
    """

    def __init__(self,
                 monitored_metrics_names,
                 min_delta=0,
                 patience=0,
                 mode='auto',
                 baseline=None,
                 ):

        #Check types
        error_msg = ("The monitored_metrics must be a dictionary!")
        assert isinstance(monitored_metrics_names,list), error_msg
        monitored_metrics_dict = dict([(x,None) for x in
                                       monitored_metrics_names])

        error_msg = ("The min_delta must be an float or dictionary")
        assert isinstance(min_delta, (dict,float)), error_msg

        error_msg = ("The mode must be a string or dictionary")
        assert isinstance(mode, (dict,str)), error_msg

        error_msg = ("The baseline must be None or a dictionary")
        assert isinstance(baseline, (dict,type(None))),error_msg


        #Initaliaze ops to correct types
        min_delta = dict.fromkeys(monitored_metrics_dict,min_delta) if (
            isinstance(min_delta,float)) else min_delta


        mode = dict.fromkeys(monitored_metrics_dict,mode) if (
            isinstance(mode,str)) else mode

        self.monitored_metrics_dict = monitored_metrics_dict
        self.baseline = baseline
        self.patience = patience
        self.min_delta = min_delta
        self.wait = dict.fromkeys(monitored_metrics_dict,0)
        self.stopped_epoch = 0
        self.monitor_op_dict = dict.fromkeys(monitored_metrics_dict,None)
        self.best = dict.fromkeys(monitored_metrics_dict,None)

        #Set up compaators and min deltas
        for mode_key,mode_val in mode.items():
            if mode_val not in ['auto', 'min', 'max']:
                warnings.warn('EarlyStopping mode %s is unknown, '
                              'fallback to auto mode.' % mode,
                              RuntimeWarning)
                mode_val = 'auto'

            if mode_val == 'min':
                self.monitor_op_dict[mode_key] = np.less
            elif mode_val == 'max':
                self.monitor_op_dict[mode_key] = np.greater
            else:
                if 'acc' in mode_key:
                    self.monitor_op_dict[mode_key] = np.greater
                else:
                    self.monitor_op_dict[mode_key] = np.less

            if self.monitor_op_dict[mode_key] == np.greater:
                self.min_delta[mode_key] *= 1
            else:
                self.min_delta[mode_key] *= -1

        #Set up baselines if necessary
        if self.baseline is not None:
            self.best = self.baseline
        else:
            for best_key,_ in self.best.items():
                starting_val = np.Inf if self.monitor_op_dict[best_key] == np.less else -np.Inf
                self.best[best_key] = starting_val

    def early_stopping_check(self, current_monitored_values):
        '''
        Arguments:
            current_monitored_values (dict) : A dictionary containing
            the most recent values corresponding to each key in
            monitored_metrics_names.
        Returns:
            stop_training (bool) : Returns True if training should stop
            early_stopping_metric_key (str) : Returns the key of the metric
            that failed to meet its continuation criterion. Returns None if
            training should continue.
            
        Rasies:
            ValueError: An empty dictionary was passed to function.
            AssertionError: `current_monitored_values` dictionary has different
            keys than those listed in `monitored_metrics_names` at class
            creation.
        '''

        if len(current_monitored_values) == 0:
            raise ValueError("Monitored Value cannot be None")

        error_msg = "Must provide current values for all monitored metrics"
        assert (self.monitored_metrics_dict.keys()
                == current_monitored_values.keys()), error_msg

        for metric_key, metric_value in current_monitored_values.items():
            if self.monitor_op_dict[metric_key](metric_value,
                                                (self.best[metric_key] + self.min_delta[metric_key])):
                self.best[metric_key] = metric_value
                self.wait[metric_key] = 0
            else:
                self.wait[metric_key] += 1
                if self.wait[metric_key] >= self.patience:
                    return True, metric_key
        return False, None
