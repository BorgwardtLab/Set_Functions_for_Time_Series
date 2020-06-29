"""Phased LSTM implementation based on the version in tensorflow contrib.

See: https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1915-L2064

Due to restructurings in tensorflow some adaptions were required. This
implementation does not use global naming of variables and thus is compatible
with the new keras style paradime.
"""

from collections.abc import Sequence
from collections import namedtuple

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.nn.rnn_cell import LSTMStateTuple

from .delta_t_utils import get_delta_t


PhasedLSTMInput = namedtuple('PhasedLSTMInput', ['times', 'x'])


def _random_exp_initializer(minval, maxval, seed=None, dtype=tf.float32):
    """Return an exponential distribution initializer.

    Args:
      minval: float or a scalar float Tensor. With value > 0. Lower bound of the
          range of random values to generate.
      maxval: float or a scalar float Tensor. With value > minval. Upper bound of
          the range of random values to generate.
      seed: An integer. Used to create random seeds.
      dtype: The data type.

    Returns:
      An initializer that generates tensors with an exponential distribution.

    """
    def _initializer(shape, dtype=dtype, partition_info=None):
        del partition_info  # Unused.
        return tf.math.exp(tf.random.uniform(
            shape, tf.math.log(minval), tf.math.log(maxval), dtype, seed=seed))

    return _initializer


class PhasedLSTMCell(tf.keras.layers.Layer):
    """Phased LSTM recurrent network cell.

    https://arxiv.org/pdf/1610.09513v1.pdf
    """

    def __init__(self, num_units, use_peepholes=False, leak=0.001,
                 ratio_on=0.1, trainable_ratio_on=True, period_init_min=0.5,
                 period_init_max=1000.0):
        """Initialize the Phased LSTM cell.

        Args:
          num_units: int, The number of units in the Phased LSTM cell.
          use_peepholes: bool, set True to enable peephole connections.
          leak: float or scalar float Tensor with value in [0, 1]. Leak applied
              during training.
          ratio_on: float or scalar float Tensor with value in [0, 1]. Ratio of
              the period during which the gates are open.
          trainable_ratio_on: bool, weather ratio_on is trainable.
          period_init_min: float or scalar float Tensor. With value > 0.
              Minimum value of the initialized period.
              The period values are initialized by drawing from the
              distribution: e^U(log(period_init_min), log(period_init_max))
              Where U(.,.) is the uniform distribution.
          period_init_max: float or scalar float Tensor.
              With value > period_init_min. Maximum value of the initialized
              period.

        """
        super().__init__()
        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._leak = leak
        self._ratio_on = ratio_on
        self._trainable_ratio_on = trainable_ratio_on
        self._period_init_min = period_init_min
        self._period_init_max = period_init_max
        self.linear1 = Dense(
            2 * self._num_units, use_bias=True, activation='sigmoid',
            name='MaskGates')
        self.linear2 = Dense(
            self._num_units, use_bias=True, activation='tanh')
        self.linear3 = Dense(
            self._num_units, use_bias=True, activation='sigmoid')

        self.period = self.add_weight(
            'period', shape=[self._num_units],
            initializer=_random_exp_initializer(
                self._period_init_min, self._period_init_max))
        self.phase = self.add_weight(
            'phase', shape=[self._num_units],
            initializer=tf.initializers.random_uniform(0., self.period.initial_value))
        self.ratio_on = self.add_weight(
            "ratio_on", [self._num_units],
            initializer=tf.constant_initializer(self._ratio_on),
            trainable=self._trainable_ratio_on)

    def build(self, input_shapes):
        time_shape, x_shape = input_shapes.times, input_shapes.x
        x_dim = x_shape[-1]

        if self._use_peepholes:
            mask_gate_and_ouput_gate_dims = 2 * self._num_units + x_dim
        else:
            mask_gate_and_ouput_gate_dims = self._num_units + x_dim

        self.linear1.build((time_shape[0], mask_gate_and_ouput_gate_dims))
        self.linear2.build((time_shape[0], self._num_units + x_dim))
        self.linear3.build((time_shape[0], mask_gate_and_ouput_gate_dims))
        super().build(input_shapes)

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def _mod(self, x, y):
        """Modulo function that propagates x gradients."""
        return tf.stop_gradient(tf.math.mod(x, y) - x) + x

    def _get_cycle_ratio(self, time):
        """Compute the cycle ratio in the dtype of the time."""
        phase = tf.cast(self.phase, dtype=time.dtype)
        period = tf.cast(self.period, dtype=time.dtype)
        shifted_time = time - phase
        cycle_ratio = self._mod(shifted_time, period) / period
        return tf.cast(cycle_ratio, dtype=tf.float32)

    def call(self, inputs, state):
        """Phased LSTM Cell.

        Args:
          inputs: A tuple of 2 Tensor.
             The first Tensor has shape [batch, 1], and type float32 or float64.
             It stores the time.
             The second Tensor has shape [batch, features_size], and type float32.
             It stores the features.
          state: rnn_cell_impl.LSTMStateTuple, state from previous timestep.
        Returns:
          A tuple containing:
          - A Tensor of float32, and shape [batch_size, num_units], representing the
            output of the cell.
          - A rnn_cell_impl.LSTMStateTuple, containing 2 Tensors of float32, shape
            [batch_size, num_units], representing the new state and the output.
        """
        (c_prev, h_prev) = state
        time, x = inputs.times, inputs.x

        if self._use_peepholes:
            input_mask_and_output_gate = tf.concat(
                [x, h_prev, c_prev], axis=-1)
        else:
            input_mask_and_output_gate = tf.concat([x, h_prev], axis=-1)

        mask_gates = self.linear1(input_mask_and_output_gate)

        input_gate, forget_gate = tf.split(
            mask_gates, axis=1, num_or_size_splits=2)

        new_input = self.linear2(tf.concat([x, h_prev], axis=-1))

        new_c = (c_prev * forget_gate + input_gate * new_input)

        output_gate = self.linear3(input_mask_and_output_gate)

        new_h = tf.tanh(new_c) * output_gate

        cycle_ratio = self._get_cycle_ratio(time)
        k_up = 2 * cycle_ratio / self.ratio_on
        k_down = 2 - k_up
        k_closed = self._leak * cycle_ratio

        k = tf.where(cycle_ratio < self.ratio_on, k_down, k_closed)
        k = tf.where(cycle_ratio < 0.5 * self.ratio_on, k_up, k)

        new_c = k * new_c + (1 - k) * c_prev
        new_h = k * new_h + (1 - k) * h_prev

        new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
        return new_h, new_state


class PhasedLSTMModel(tf.keras.Model):
    def __init__(self, output_activation, output_dims, n_units, use_peepholes,
                 leak, period_init_max):
        self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }
        super().__init__()
        self.demo_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_units, activation='relu'),
                tf.keras.layers.Dense(2*n_units)
            ],
            name='demo_encoder'
        )
        if isinstance(output_dims, Sequence):
            # We have an online prediction scenario
            assert output_dims[0] is None
            self.return_sequences = True
            output_dims = output_dims[1]
        else:
            self.return_sequences = False
        self.rnn = tf.keras.layers.RNN(
            PhasedLSTMCell(
                n_units, use_peepholes=use_peepholes,
                leak=leak, period_init_max=period_init_max
            ),
            return_sequences=self.return_sequences
        )
        self.output_layer = tf.keras.layers.Dense(
            output_dims, activation=output_activation)

    def call(self, inputs):
        demo, times, values, measurements, dt, lengths = inputs
        demo_encoded = self.demo_encoder(demo)
        initial_state = LSTMStateTuple(*tf.split(demo_encoded, 2, axis=-1))

        values = tf.concat(
            (values, tf.cast(measurements, tf.float32), dt), axis=-1)
        mask = tf.sequence_mask(tf.squeeze(lengths, axis=-1), name='mask')
        out = self.rnn(
            PhasedLSTMInput(times=times, x=values),
            mask=mask,
            initial_state=initial_state
        )
        return self.output_layer(out)

    def data_preprocessing_fn(self):
        def add_delta_t_tensor(ts, label):
            demo, times, values, measurement_indicators, length = ts
            times = tf.expand_dims(times, -1)
            dt = get_delta_t(times, values, measurement_indicators)
            return (
                (demo, times, values, measurement_indicators, dt, length),
                label
            )
        return add_delta_t_tensor

    @classmethod
    def get_hyperparameters(cls):
        import tensorboard.plugins.hparams.api as hp
        from ..training_utils import HParamWithDefault
        return [
            HParamWithDefault(
                'n_units', hp.Discrete([32, 64, 128, 256, 512, 1024]),
                default=32),
            HParamWithDefault(
                'use_peepholes', hp.Discrete([True, False]), default=False),
            HParamWithDefault(
                'leak', hp.Discrete([0.001, 0.005, 0.01]), default=0.001),
            HParamWithDefault(
                'period_init_max', hp.Discrete([10., 100., 1000.]),
                default=1000.)
        ]

    @classmethod
    def from_hyperparameter_dict(cls, task, hparams):
        return cls(
            output_activation=task.output_activation,
            output_dims=task.n_outputs,
            n_units=hparams['n_units'],
            use_peepholes=hparams['use_peepholes'],
            leak=hparams['leak'],
            period_init_max=hparams['period_init_max']
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return self._config
