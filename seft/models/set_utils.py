import inspect
from itertools import chain
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout


class PaddedToSegments(tf.keras.layers.Layer):
    """Convert a padded tensor with mask to a stacked tensor with segments."""

    def compute_output_shape(self, input_shape):
        return (None, input_shape[-1])

    def call(self, inputs, mask):
        valid_observations = tf.where(mask)
        collected_values = tf.gather_nd(inputs, valid_observations)
        return collected_values, valid_observations[:, 0]


def cumulative_softmax_weighting(values, preattention, segment_ids, eps=1e-7):
    """Cumulative softmax weighting of values.

    Args:
        values: Values expected shape [n_samples, feature_dim]
        preattention: Preattention values, expected shape [n_samples, n_heads]
        segment_ids: Segment ids

    Returns:
    """
    head_preattn = tf.unstack(preattention, axis=-1)
    exp_head_preattn = []
    cumulative_exp_preattn = []

    for cur_head_preattn in head_preattn:
        # For numerical stability subtract the max from data values
        max_values = tf.math.segment_max(cur_head_preattn, segment_ids)
        max_values = tf.gather_nd(max_values, tf.expand_dims(segment_ids, -1))
        max_values = tf.stop_gradient(max_values)

        normalized = cur_head_preattn - max_values
        exp_preattn = tf.exp(normalized, name='exp_preattn')
        exp_head_preattn.append(exp_preattn)
        cumulative_exp_preattn.append(
            cumulative_segment_sum(
                exp_preattn, segment_ids, name='segment_cumsum'))

    exp_head_preattn = tf.stack(exp_head_preattn, -1)
    weighted_values = \
        tf.expand_dims(values, 1) * tf.expand_dims(exp_head_preattn, -1)

    cumulative_exp_preattn = tf.stack(cumulative_exp_preattn, axis=-1)

    # Sum the values
    out = (
        (cumulative_segment_sum(weighted_values, segment_ids) + eps)
        / (tf.expand_dims(cumulative_exp_preattn, -1) + eps)
    )
    return out


def cumulative_segment_wrapper(fun):
    """Wrap a cumulative function such that it can be applied to segments.

    Args:
        fun: The cumulative function

    Returns:
        Wrapped function.

    """
    def wrapped_segment_op(x, segment_ids, **kwargs):
        with tf.compat.v1.name_scope(
                None, default_name=fun.__name__+'_segment_wrapper', values=[x]):
            segments, _ = tf.unique(segment_ids)
            n_segments = tf.shape(segments)[0]
            output_array = tf.TensorArray(
                x.dtype, size=n_segments, infer_shape=False)

            def loop_cond(i, out):
                return i < n_segments

            def execute_cumulative_op_on_segment(i, out):
                segment_indices = tf.where(tf.equal(segment_ids, segments[i]))
                seg_begin = tf.reduce_min(segment_indices)
                seg_end = tf.reduce_max(segment_indices)
                segment_data = x[seg_begin:seg_end+1]
                out = out.write(i, fun(segment_data, **kwargs))
                return i+1, out

            i_end, filled_array = tf.while_loop(
                loop_cond,
                execute_cumulative_op_on_segment,
                loop_vars=(tf.constant(0), output_array),
                parallel_iterations=10,
                swap_memory=True
            )
            output_tensor = filled_array.concat()
            output_tensor.set_shape(x.get_shape())
            return output_tensor

    return wrapped_segment_op


def cumulative_mean(tensor):
    """Cumulative mean of a rank 2 tensor.

    Args:
        tensor: Input tensor

    Returns:
        Tensor with same shape as input but containing cumulative mean.

    """
    assert len(tensor.shape) == 2
    n_elements = tf.cast(tf.shape(tensor)[0], tensor.dtype)
    start = tf.constant(1, dtype=tensor.dtype)
    n_elements_summed = tf.range(start, n_elements+1, dtype=tensor.dtype)
    return tf.cumsum(tensor, axis=0) / tf.expand_dims(n_elements_summed, -1)


cumulative_segment_mean = cumulative_segment_wrapper(cumulative_mean)
cumulative_segment_sum = cumulative_segment_wrapper(tf.math.cumsum)


def cumulative_softmax(tensor):
    """Cumulative softmax operation

    Args:
        tensor: 2d tensor

    Returns:
    """
    assert len(tensor.shape) == 1
    max_values = tf.reduce_max(tensor, axis=0)
    normalized = tensor - max_values

    numerator = tf.exp(normalized)
    denominator = tf.cumsum(numerator, axis=0)
    return numerator / denominator


class SegmentLayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['epsilon'] = self.epsilon
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        dim = input_shape[-1]
        self.gain = self.add_weight(
            name='gain',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, segment_ids, **kwargs):
        # Get one mean per instance
        segments, _, count = tf.unique_with_counts(segment_ids)
        divisor = tf.cast(count * inputs.get_shape()[-1], tf.float32)
        mean = tf.reduce_sum(
            tf.math.segment_sum(inputs, segment_ids),
            axis=-1
        ) / divisor
        mean = tf.gather(mean, segment_ids, axis=-1)[:, None]

        variance = tf.reduce_sum(
            tf.math.segment_sum((inputs - mean) ** 2, segment_ids),
            axis=-1
        ) / divisor
        variance = tf.gather(variance, segment_ids, axis=-1)[:, None]

        epsilon = tf.constant(self.epsilon, dtype=tf.float32)
        normalized_inputs = (inputs - mean) / tf.math.sqrt(variance + epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result


class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        self.axis = axis
        self.epsilon = epsilon
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['axis'] = self.axis
        config['epsilon'] = self.epsilon
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        dim = input_shape[-1]
        self.gain = self.add_weight(
            name='gain',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        mean = tf.reduce_mean(inputs, axis=self.axis, keepdims=True)
        variance = tf.reduce_mean(
            (inputs - mean) ** 2, axis=self.axis, keepdims=True)
        epsilon = tf.constant(self.epsilon, dtype=tf.float32)
        normalized_inputs = (inputs - mean) / tf.sqrt(variance + epsilon)
        result = self.gain * normalized_inputs + self.bias
        return result


class SegmentAggregation(tf.keras.layers.Layer):
    def __init__(self, aggregation_fn='sum', cumulative=False):
        super().__init__()
        self.cumulative = cumulative
        self.aggregation_fn = self._get_aggregation_fn(aggregation_fn)

    def _get_aggregation_fn(self, aggregation_fn):
        if not self.cumulative:
            if aggregation_fn == 'sum':
                return tf.math.segment_sum
            elif aggregation_fn == 'mean':
                return tf.math.segment_mean
            elif aggregation_fn == 'max':
                return tf.math.segment_max
            else:
                raise ValueError('Invalid aggregation function')
        else:
            if aggregation_fn == 'sum':
                return cumulative_segment_wrapper(tf.math.cumsum)
            elif aggregation_fn == 'mean':
                return cumulative_segment_wrapper(cumulative_mean)
            elif aggregation_fn == 'max':
                raise ValueError('max aggregation function not supported with cumulative aggregation.')
            else:
                raise ValueError('Invalid aggregation function')

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, data, segment_ids):
        assert segment_ids is not None
        return self.aggregation_fn(data, segment_ids)


class MySequential(tf.keras.layers.Layer):
    """Simplified version of tf.keras.Sequential, supports segment_ids."""

    def __init__(self, layers, **kwargs):
        super().__init__(**kwargs)
        self.layers = []
        for layer in layers:
            self.add(layer)

    def compute_output_shape(self, input_shape):
        last_shape = input_shape
        for layer in self.layers:
            last_shape = layer.compute_output_shape(last_shape)
        return last_shape

    def build(self, input_shape):
        last_shape = input_shape
        for layer in self.layers:
            layer.build(last_shape)
            last_shape = layer.compute_output_shape(last_shape)
        super().build(input_shape)

    def add(self, layer):
        next_layer_index = len(self.layers)
        layer_name = f'{next_layer_index}_{layer.name}'
        self.layers.append(layer)
        setattr(self, layer_name, layer)
        self.built = False

    def call(self, inputs, segment_ids):
        outputs = inputs  # handle the corner case where self.layers is empty
        for layer in self.layers:
            # During each iteration, `inputs` are the inputs to `layer`, and `outputs`
            # are the outputs of `layer` applied to `inputs`. At the end of each
            # iteration `inputs` is set to `outputs` to prepare for the next layer.
            kwargs = {}
            argspec = inspect.getargspec(layer.call)
            if 'segment_ids' in argspec.args:
                kwargs['segment_ids'] = segment_ids

            outputs = layer(inputs, **kwargs)

            # `outputs` will be the inputs to the next layer.
            inputs = outputs

        return outputs


def build_dense_dropout_model(n_layers, width, dropout, dense_kwargs,
                              name=None):
    """Build a Sequential model composed of stacked Dense and Dropout blocks.

    Calling with n_layers=1 corresponds to the output:
    Sequential([Dense(width), Dropout(dropout)])

    Args:
        n_layers: Number of layers to stack
        width: Width of the layers
        dropout: Dropout probability
        dense_kwargs: Additionaly kwargs for the Dense class

    Returns:
        Sequential model of stacked Dense Dropout layers

    """
    if dropout > 0.:
        layers = list(chain(*(
            (Dense(width, **dense_kwargs), Dropout(dropout))
            for i in range(n_layers)
        )))
    else:
        layers = [Dense(width, **dense_kwargs) for i in range(n_layers)]
    return tf.keras.Sequential(layers, name=name)
