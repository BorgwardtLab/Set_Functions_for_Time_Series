"""GRU_simple.

RECURRENT NEURAL NETWORKS FOR MULTIVARIATE TIME SERIES WITH MISSING VALUES,
Che et al., 2016
"""
from collections.abc import Sequence
import tensorflow as tf

from .delta_t_utils import get_delta_t


class GRUSimpleModel(tf.keras.Model):
    def __init__(self, output_activation, output_dims, n_units,
                 dropout, recurrent_dropout):
        self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }
        super().__init__()
        self.demo_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_units, activation='relu'),
                tf.keras.layers.Dense(n_units)
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

        self.rnn = tf.keras.layers.GRU(
            n_units, dropout=dropout, recurrent_dropout=recurrent_dropout,
            return_sequences=self.return_sequences)
        self.output_layer = tf.keras.layers.Dense(
            output_dims, activation=output_activation)

    def call(self, inputs):
        demo, values, measurements, dt, lengths = inputs
        demo_encoding = self.demo_encoder(demo)
        values = tf.concat(
            (values, tf.cast(measurements, tf.float32), dt), axis=-1)
        mask = tf.sequence_mask(tf.squeeze(lengths, axis=-1), name='mask')
        out = self.rnn(
            values, mask=mask, initial_state=demo_encoding)
        return self.output_layer(out)

    def data_preprocessing_fn(self):
        def add_delta_t_tensor(ts, label):
            demo, times, values, measurement_indicators, length = ts
            times = tf.expand_dims(times, -1)
            dt = get_delta_t(times, values, measurement_indicators)
            return (demo, values, measurement_indicators, dt, length), label
        return add_delta_t_tensor

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

