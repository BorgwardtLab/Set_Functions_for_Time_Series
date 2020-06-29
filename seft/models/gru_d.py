"""Implementation of GRU-D model.

The below implementation is based on and adapted from
https://github.com/PeterChe1990/GRU-D

Which is published unter the MIT licence.
"""
from collections.abc import Sequence
from collections import namedtuple
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.python.keras.layers.recurrent import _generate_dropout_mask
from tensorflow.python.keras.layers.recurrent import GRUCell
from tensorflow.python.keras.utils.generic_utils import (
    serialize_keras_object, custom_object_scope)


GRUDInput = namedtuple('GRUDInput', ['values', 'mask', 'times'])
GRUDState = namedtuple('GRUDState', ['h', 'x_keep', 's_prev'])

__all__ = ['exp_relu', 'get_activation']

_SUPPORTED_IMPUTATION = ['zero', 'forward', 'raw']


def exp_relu(x):
    return K.exp(-K.relu(x))


def get_activation(identifier):
    if identifier is None:
        return None
    with custom_object_scope({'exp_relu': exp_relu}):
        return tf.keras.activations.get(identifier)


class GRUDCell(GRUCell):
    """Cell class for the GRU-D layer. An extension of `GRUCell`.
    Notice: Calling with only 1 tensor due to the limitation of Keras.
    Building, computing the shape with the input_shape as a list of length 3.
    # TODO: dynamic imputation
    """

    def __init__(self, units, x_imputation='zero', input_decay='exp_relu',
                 hidden_decay='exp_relu', use_decay_bias=True,
                 feed_masking=True, masking_decay=None,
                 decay_initializer='zeros', decay_regularizer=None,
                 decay_constraint=None, **kwargs):
        assert 'reset_after' not in kwargs or not kwargs['reset_after'], (
            'Only the default GRU reset gate can be used in GRU-D.'
        )
        assert ('implementation' not in kwargs
                or kwargs['implementation'] == 1), (
                    'Only Implementation-1 (larger number of smaller operations) '
                    'is supported in GRU-D.'
                )

        assert x_imputation in _SUPPORTED_IMPUTATION, (
            'x_imputation {} argument is not supported.'.format(x_imputation)
        )
        self.x_imputation = x_imputation

        self.input_decay = get_activation(input_decay)
        self.hidden_decay = get_activation(hidden_decay)
        self.use_decay_bias = use_decay_bias

        assert (feed_masking or masking_decay is None
                or masking_decay == 'None'), (
                    'Mask needs to be fed into GRU-D to enable the mask_decay.'
                )
        self.feed_masking = feed_masking
        if self.feed_masking:
            self.masking_decay = get_activation(masking_decay)
            self._masking_dropout_mask = None
        else:
            self.masking_decay = None

        if (self.input_decay is not None
            or self.hidden_decay is not None
            or self.masking_decay is not None):
            self.decay_initializer = initializers.get(decay_initializer)
            self.decay_regularizer = regularizers.get(decay_regularizer)
            self.decay_constraint = constraints.get(decay_constraint)

        self._input_dim = None
        # We need to wrap a try arround this as GRUCell sets state_size
        try:
            super().__init__(units, **kwargs)
        except AttributeError:
            pass

    @property
    def state_size(self):
        return GRUDState(
            h=self.units, x_keep=self._input_dim, s_prev=self._input_dim)

    def get_initial_state(self, inputs, batch_size, dtype):
        if inputs is None:
            return GRUDState(
                tf.zeros(tf.stack([batch_size, self.units])),
                tf.zeros(tf.stack([batch_size, self._input_dim])),
                tf.zeros(tf.stack([batch_size, self._input_dim]))
            )
        else:
            if self.go_backwards:
                return GRUDState(
                    tf.zeros(tf.stack([batch_size, self.units])),
                    tf.zeros(tf.stack([batch_size, self._input_dim])),
                    tf.tile(
                        tf.reduce_max(inputs.times, axis=1),
                        [1, self._input_dim]
                    )
                )
            else:
                return GRUDState(
                    tf.zeros(tf.stack([batch_size, self.units])),
                    tf.zeros(tf.stack([batch_size, self._input_dim])),
                    tf.tile(inputs.times[:, 0, :], [1, self._input_dim])
                )

    def build(self, input_shape):
        """
        Args:
            input_shape: A tuple of 3 shapes (from x, m, s, respectively)
        """
        self._input_dim = input_shape.values[-1]
        # Validate the shape of the input first. Borrow the idea from `_Merge`.
        assert len(input_shape.times) == 2

        # Borrow the logic from GRUCell for the same part.
        super(GRUDCell, self).build(input_shape.values)

        # Implementation of GRUCell changed, split the tensors here so we dont
        # need to rewrite the code
        self.kernel_z, self.kernel_r, self.kernel_h = tf.split(
            self.kernel, 3, axis=-1)
        (self.recurrent_kernel_z,
         self.recurrent_kernel_r,
         self.recurrent_kernel_h) = tf.split(self.recurrent_kernel, 3, axis=-1)
        (self.input_bias_z,
         self.input_bias_r,
         self.input_bias_h) = tf.split(self.bias, 3, axis=-1)

        # Build the own part of GRU-D.
        if self.input_decay is not None:
            self.input_decay_kernel = self.add_weight(
                shape=(self._input_dim,),
                name='input_decay_kernel',
                initializer=self.decay_initializer,
                regularizer=self.decay_regularizer,
                constraint=self.decay_constraint
            )
            if self.use_decay_bias:
                self.input_decay_bias = self.add_weight(
                    shape=(self._input_dim,),
                    name='input_decay_bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint
                )
        if self.hidden_decay is not None:
            self.hidden_decay_kernel = self.add_weight(
                shape=(self._input_dim, self.units),
                name='hidden_decay_kernel',
                initializer=self.decay_initializer,
                regularizer=self.decay_regularizer,
                constraint=self.decay_constraint
            )
            if self.use_decay_bias:
                self.hidden_decay_bias = self.add_weight(
                    shape=(self.units,),
                    name='hidden_decay_bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint
                )
        if self.feed_masking:
            self.masking_kernel = self.add_weight(
                shape=(self._input_dim, self.units * 3),
                name='masking_kernel',
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint
            )
            if self.masking_decay is not None:
                self.masking_decay_kernel = self.add_weight(
                    shape=(self._input_dim,),
                    name='masking_decay_kernel',
                    initializer=self.decay_initializer,
                    regularizer=self.decay_regularizer,
                    constraint=self.decay_constraint
                )
                if self.use_decay_bias:
                    self.masking_decay_bias = self.add_weight(
                        shape=(self._input_dim,),
                        name='masking_decay_bias',
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint
                    )
            (
                self.masking_kernel_z,
                self.masking_kernel_r,
                self.masking_kernel_h
            ) = tf.split(self.masking_kernel, 3, axis=-1)
        self.built = True

    def reset_masking_dropout_mask(self):
        self._masking_dropout_mask = None

    def call(self, inputs, states, training=None):
        """We need to reimplmenet `call` entirely rather than reusing that
        from `GRUCell` since there are lots of differences.
        Args:
            inputs: One tensor which is stacked by 3 inputs (x, m, s)
                x and m are of shape (n_batch * input_dim).
                s is of shape (n_batch, 1).
            states: states and other values from the previous step.
                (h_tm1, x_keep_tm1, s_prev_tm1)
        """
        # Get inputs and states
        input_x = inputs.values
        input_m = inputs.mask
        input_s = inputs.times

        h_tm1, x_keep_tm1, s_prev_tm1 = states
        # previous memory ([n_batch * self.units])
        # previous input x ([n_batch * input_dim])
        # and the subtraction term (of delta_t^d in Equation (2))
        # ([n_batch * input_dim])
        input_1m = 1. - tf.cast(input_m, tf.float32)
        input_d = input_s - s_prev_tm1

        dp_mask = self.get_dropout_mask_for_cell(
                input_x, training, count=3) 
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
                h_tm1, training, count=3)

        if self.feed_masking:
            if 0. < self.dropout < 1. and self._masking_dropout_mask is None:
                self._masking_dropout_mask = _generate_dropout_mask(
                    tf.ones_like(input_m, dtype=tf.float32),
                    self.dropout,
                    training=training,
                    count=3)
            m_dp_mask = self._masking_dropout_mask

        # Compute decay if any
        if self.input_decay is not None:
            gamma_di = input_d * self.input_decay_kernel
            if self.use_decay_bias:
                gamma_di = K.bias_add(gamma_di, self.input_decay_bias)
            gamma_di = self.input_decay(gamma_di)
        if self.hidden_decay is not None:
            gamma_dh = K.dot(input_d, self.hidden_decay_kernel)
            if self.use_decay_bias:
                gamma_dh = K.bias_add(gamma_dh, self.hidden_decay_bias)
            gamma_dh = self.hidden_decay(gamma_dh)
        if self.feed_masking and self.masking_decay is not None:
            gamma_dm = input_d * self.masking_decay_kernel
            if self.use_decay_bias:
                gamma_dm = K.bias_add(gamma_dm, self.masking_decay_bias)
            gamma_dm = self.masking_decay(gamma_dm)

        # Get the imputed or decayed input if needed
        # and `x_keep_t` for the next time step

        if self.input_decay is not None:
            x_keep_t = tf.where(input_m, input_x, x_keep_tm1)
            x_t = tf.where(input_m, input_x, gamma_di * x_keep_t)
        elif self.x_imputation == 'forward':
            x_t = tf.where(input_m, input_x, x_keep_tm1)
            x_keep_t = x_t
        elif self.x_imputation == 'zero':
            x_t = tf.where(input_m, input_x, K.zeros_like(input_x))
            x_keep_t = x_t
        elif self.x_imputation == 'raw':
            x_t = input_x
            x_keep_t = x_t
        else:
            raise ValueError('No input decay or invalid x_imputation '
                             '{}.'.format(self.x_imputation))

        # Get decayed hidden if needed
        if self.hidden_decay is not None:
            h_tm1d = gamma_dh * h_tm1
        else:
            h_tm1d = h_tm1

        # Get decayed masking if needed
        if self.feed_masking:
            m_t = input_1m
            if self.masking_decay is not None:
                m_t = gamma_dm * m_t

        # Apply the dropout
        if 0. < self.dropout < 1.:
            x_z, x_r, x_h = x_t * dp_mask[0], x_t * dp_mask[1], x_t * dp_mask[2]
            if self.feed_masking:
                m_z, m_r, m_h = (m_t * m_dp_mask[0],
                                 m_t * m_dp_mask[1],
                                 m_t * m_dp_mask[2]
                                )
        else:
            x_z, x_r, x_h = x_t, x_t, x_t
            if self.feed_masking:
                m_z, m_r, m_h = m_t, m_t, m_t
        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z, h_tm1_r = (h_tm1d * rec_dp_mask[0],
                                         h_tm1d * rec_dp_mask[1],
                                        )
        else:
            h_tm1_z, h_tm1_r = h_tm1d, h_tm1d

        # Get z_t, r_t, hh_t
        z_t = K.dot(x_z, self.kernel_z) + K.dot(h_tm1_z, self.recurrent_kernel_z)
        r_t = K.dot(x_r, self.kernel_r) + K.dot(h_tm1_r, self.recurrent_kernel_r)
        hh_t = K.dot(x_h, self.kernel_h)
        if self.feed_masking:
            z_t += K.dot(m_z, self.masking_kernel_z)
            r_t += K.dot(m_r, self.masking_kernel_r)
            hh_t += K.dot(m_h, self.masking_kernel_h)
        if self.use_bias:
            z_t = K.bias_add(z_t, self.input_bias_z)
            r_t = K.bias_add(r_t, self.input_bias_r)
            hh_t = K.bias_add(hh_t, self.input_bias_h)
        z_t = self.recurrent_activation(z_t)
        r_t = self.recurrent_activation(r_t)

        if 0. < self.recurrent_dropout < 1.:
            h_tm1_h = r_t * h_tm1d * rec_dp_mask[2]
        else:
            h_tm1_h = r_t * h_tm1d
        hh_t = self.activation(hh_t + K.dot(h_tm1_h, self.recurrent_kernel_h))

        # get h_t
        h_t = z_t * h_tm1 + (1 - z_t) * hh_t

        # get s_prev_t
        s_prev_t = tf.where(input_m,
                            K.tile(input_s, [1, self.state_size[-1]]),
                            s_prev_tm1)
        return h_t, GRUDState(h_t, x_keep_t, s_prev_t)

    def get_config(self):
        # Remember to record all args of the `__init__`
        # which are not covered by `GRUCell`.
        config = {'x_imputation': self.x_imputation,
                  'input_decay': serialize_keras_object(self.input_decay),
                  'hidden_decay': serialize_keras_object(self.hidden_decay),
                  'use_decay_bias': self.use_decay_bias,
                  'feed_masking': self.feed_masking,
                  'masking_decay': serialize_keras_object(self.masking_decay),
                  'decay_initializer': initializers.serialize(self.decay_initializer),
                  'decay_regularizer': regularizers.serialize(self.decay_regularizer),
                  'decay_constraint': constraints.serialize(self.decay_constraint)
                 }
        base_config = super(GRUDCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRUD(tf.keras.layers.RNN):
    def __init__(self, units, x_imputation='zero', input_decay='exp_relu',
                 hidden_decay='exp_relu', use_decay_bias=True,
                 feed_masking=True, masking_decay=None,
                 decay_initializer='zeros', decay_regularizer=None,
                 decay_constraint=None,  activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True, kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal', bias_initializer='zeros',
                 kernel_regularizer=None, recurrent_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, recurrent_constraint=None,
                 bias_constraint=None, dropout=0., recurrent_dropout=0.,
                 implementation=1, return_sequences=False, return_state=False,
                 go_backwards=False, stateful=False, unroll=False,
                 reset_after=False, **kwargs):
        cell = GRUDCell(
            units=units,
            x_imputation=x_imputation,
            input_decay=input_decay,
            hidden_decay=hidden_decay,
            use_decay_bias=use_decay_bias,
            feed_masking=feed_masking,
            masking_decay=masking_decay,
            decay_initializer=decay_initializer,
            decay_regularizer=decay_regularizer,
            decay_constraint=decay_constraint,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            reset_after=reset_after,
            dtype=kwargs.get('dtype')
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs
        )
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()
        self.cell.reset_masking_dropout_mask()
        return super().call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    def get_config(self):
        config = {
            'units':
                self.units,
            'x_imputation': self.x_imputation,
            'input_decay': serialize_keras_object(self.input_decay),
            'hidden_decay': serialize_keras_object(self.hidden_decay),
            'use_decay_bias': self.use_decay_bias,
            'feed_masking': self.feed_masking,
            'masking_decay': serialize_keras_object(self.masking_decay),
            'decay_initializer': initializers.get(self.decay_initializer),
            'decay_regularizer': regularizers.get(self.decay_regularizer),
            'decay_constraint': constraints.get(self.decay_constraint),
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout,
            'implementation':
                self.implementation,
            'reset_after':
                self.reset_after
        }
        base_config = super().get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))


class GRUDModel(tf.keras.Model):
    def __init__(self, output_activation, output_dims, n_units, dropout,
                 recurrent_dropout):
        self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }
        super().__init__()
        self.n_units = n_units
        if isinstance(output_dims, Sequence):
            # We have an online prediction scenario
            assert output_dims[0] is None
            self.return_sequences = True
            output_dims = output_dims[1]
        else:
            self.return_sequences = False
        self.rnn = GRUD(
            n_units, dropout=dropout, recurrent_dropout=recurrent_dropout,
            return_sequences=self.return_sequences
        )
        self.output_layer = tf.keras.layers.Dense(
            output_dims, activation=output_activation)

    def build(self, input_shape):
        demo, times, values, measurements, lengths = input_shape
        self.rnn.build(
            GRUDInput(values=values, mask=measurements, times=times + (1,)))
        self.demo_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.n_units, activation='relu'),
                tf.keras.layers.Dense(self.rnn.cell.state_size[0])
            ],
            name='demo_encoder'
        )
        self.demo_encoder.build(demo)

    def call(self, inputs):
        demo, times, values, measurements, lengths = inputs
        times = tf.expand_dims(times, -1)

        demo_encoded = self.demo_encoder(demo)
        initial_state = GRUDState(
            demo_encoded,
            tf.zeros(tf.stack([tf.shape(demo)[0], self.rnn.cell._input_dim])),
            tf.tile(times[:, 0, :], [1, self.rnn.cell._input_dim])
        )

        mask = tf.sequence_mask(tf.squeeze(lengths, axis=-1), name='mask')
        grud_output = self.rnn(
            GRUDInput(
                values=values,
                mask=measurements,
                times=times
            ),
            mask=mask,
            initial_state=initial_state
        )
        return self.output_layer(grud_output)

    def data_preprocessing_fn(self):
        return None

    @classmethod
    def get_hyperparameters(cls):
        from ..training_utils import HParamWithDefault
        import tensorboard.plugins.hparams.api as hp
        return [
            HParamWithDefault(
                'n_units',
                hp.Discrete([32, 64, 128, 256, 512, 1024]),
                default=32
            ),
            HParamWithDefault(
                'dropout',
                hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4]),
                default=0.0
            ),
            HParamWithDefault(
                'recurrent_dropout',
                hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4]),
                default=0.0
            )
        ]

    @classmethod
    def from_hyperparameter_dict(cls, task, hparams):
        return cls(output_activation=task.output_activation,
                   output_dims=task.n_outputs,
                   n_units=hparams['n_units'],
                   dropout=hparams['dropout'],
                   recurrent_dropout=hparams['recurrent_dropout'])

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return self._config
