"""Module containing utility functions specific to implementing models."""
import logging

import tensorflow as tf

K = tf.keras.backend


def build_and_compute_output_shape(layer, input_shape):
    layer.build(input_shape)
    return layer.compute_output_shape(input_shape)


class ResNetBlock(tf.keras.layers.Layer):
    def __init__(self, bottleneck, **kwargs):
        super.__init__(**kwargs)
        self.bottleneck = bottleneck
        self.layer1 = tf.keras.layers.Dense(bottleneck, activation='relu')
        self.layer2 = None
        self.bn = tf.keras.layers.BatchNormalization(bottleneck)

    def build(self, input_shape):
        layer1_out = build_and_compute_output_shape(self.layer1, input_shape)
        bn_out = build_and_compute_output_shape(self.bn, layer1_out)
        layer2_out = build_and_compute_output_shape(self.layer2, bn_out)
        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape


def build_mask(data, lengths):
    """Create mask for data tensor according to lengths."""
    mask = tf.sequence_mask(lengths, maxlen=tf.shape(data)[1], dtype=tf.int32,
                            name='mask')
    return mask


# pylint: disable=missing-docstring
def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = tf.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = tf.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return tf.reduce_mean(
        tf.squared_difference(first_log, second_log), axis=-1)


def mask_observations(data, mask):
    """Mask invalid observations of data using masking tensor.

    Args:
        data: tensor (bs x observations x ...) to mask
        lengths: Masking tensor
    Returns:
        Masked tensor

    """
    # Expand dimensionality to allow broadcasing
    dims_to_add = len(data.get_shape()) - len(mask.get_shape())
    for i in range(dims_to_add):
        mask = tf.expand_dims(mask, -1)
    masked_data = data * tf.cast(mask, tf.float32)
    return masked_data


def segment_softmax(data, segment_ids, eps=1e-7):
    """Compute numerically stable softmax accroding to segments.

    Computes the softmax along the last axis of data, while grouping values
    according to their segment ids.

    Args:
        data:
        segment_ids:

    Returns:
    """
    # For numerical stability subtract the max from data values
    max_values = tf.math.segment_max(data, segment_ids)
    max_values = tf.gather_nd(max_values, tf.expand_dims(segment_ids, -1))
    max_values = tf.stop_gradient(max_values)
    normalized = data - max_values

    numerator = tf.exp(normalized)
    denominator = tf.math.segment_sum(numerator, segment_ids)
    denominator = tf.gather_nd(denominator, tf.expand_dims(segment_ids, -1))

    # Use this to avoid problems when computing the softmax, we sometime got
    # NaNs due to division by zero. If that occurs simply replace output with
    # zero instead of NaN.
    softmax = numerator / (denominator + eps)
    return softmax


def training_placeholder():
    """Either gets or creates the boolean placeholder `is_training`.

    The placeholder is initialized to have a default value of False,
    indicating that training is not taking place.
    Thus it is required to pass True to the placeholder
    to indicate training being active.

    Returns:
        tf.placeholder_with_default(False, name='is_training')
    """
    try:
        training = tf.get_default_graph().get_tensor_by_name('is_training:0')
    except KeyError:
        # We need to set this variable scope, otherwise the name of the
        # placeholder would be dependent on the variable scope of the caller
        cur_scope = tf.get_variable_scope().name
        if cur_scope == '':
            training = tf.placeholder_with_default(
                False, name='is_training', shape=[])
        else:
            with tf.variable_scope('/'):
                training = tf.placeholder_with_default(
                    False, name='is_training', shape=[])
    return training


def add_scope(fn):
    """Decorate method by wrapping it into a tensorflow name scope."""
    fn_name = fn.__name__
    if fn_name.startswith('_'):
        fn_name = fn_name[1:]

    def wrapped_fn(self, *args, **kwargs):
        # Start with a '/' to indicate absolute address
        class_name_scope = self.name
        function_name_scope = fn_name.replace('_', '-')
        with tf.name_scope(None):
            with tf.name_scope(class_name_scope+function_name_scope):
                return fn(self, *args, **kwargs)
    return wrapped_fn


def normalized_l2_regularizer(scale, scope=None):
    """Return a function that applys L2 regularization to weights.

    This implementation returns the average l2 norm (per weight) and thus
    allows defining the degree of regularization indepedent of the layer sizes.

    Args:
      scale: A scalar multiplier `Tensor`. 0.0 disables the regularizer.
      scope: An optional scope name.
    Returns:
      A function with signature `l2(weights)` that applies L2 regularization.

    """
    if scale < 0.:
        raise ValueError('Setting a scale less than 0 on a regularizer: %g.' %
                         scale)
    if scale == 0.:
        logging.info('Scale of 0 disables regularizer.')
        return lambda _: None

    def l2(weights):
        """Apply l2 regularization to weights."""
        with tf.name_scope(scope, 'norm_l2_regularizer', [weights]) as name:
            my_scale = tf.convert_to_tensor(scale,
                                            dtype=weights.dtype.base_dtype,
                                            name='scale')
            size_of_tensor = tf.cast(
                tf.reduce_prod(tf.shape(weights)),
                weights.dtype.base_dtype
            )
            return tf.multiply(
                my_scale,
                tf.nn.l2_loss(weights) / size_of_tensor,
                name=name
            )

    return l2


def pad_and_expand(tensor, maxlen):
    """Pad 1D tensor along last dim and add zeroth dimension for stacking."""
    padding_length = maxlen - tf.shape(tensor)[-1]
    padded = tf.pad(tensor, [[0, padding_length]])
    return tf.expand_dims(padded, axis=0)


def pad_and_expand2D(tensor, maxlen):
    """Pad 2D tensor along first dim and add zeroth dimension for stacking."""
    padding_length = maxlen - tf.shape(tensor)[0]
    padded = tf.pad(tensor, [[0, padding_length], [0, 0]])
    return tf.expand_dims(padded, axis=0)
