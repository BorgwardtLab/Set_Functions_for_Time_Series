"""Module with implementation of Transformer architecture."""
from collections.abc import Sequence
import tensorflow as tf
import numpy as np
import keras_transformer

from .set_utils import PaddedToSegments, SegmentAggregation


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_time=20000, n_dim=10, **kwargs):
        self.max_time = max_time
        self.n_dim = n_dim
        self._num_timescales = self.n_dim // 2
        super().__init__(**kwargs)

    def get_timescales(self):
        # This is a bit hacky, but works
        timescales = self.max_time ** np.linspace(0, 1, self._num_timescales)
        return timescales

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.timescales = self.add_weight(
            'timescales',
            (self._num_timescales, ),
            trainable=False,
            initializer=tf.keras.initializers.Constant(self.get_timescales())
        )

    def __call__(self, times):
        scaled_time = times / self.timescales[None, None, :]
        signal = tf.concat(
            [
                tf.sin(scaled_time),
                tf.cos(scaled_time)
            ],
            axis=-1)
        return signal

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_dim)


class TransformerModel(tf.keras.Model):
    def __init__(self, output_activation, output_dims, n_dims, n_heads,
                 n_layers, dropout, attn_dropout, aggregation_fn,
                 max_timescale):
        self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }
        super().__init__()
        self.positional_encoding = PositionalEncoding(
            max_timescale, n_dim=n_dims)
        self.demo_embedding = tf.keras.layers.Dense(n_dims, activation=None)
        self.element_embedding = tf.keras.layers.Dense(n_dims, activation=None)

        if isinstance(output_dims, Sequence):
            # We have an online prediction scenario
            assert output_dims[0] is None
            self.return_sequences = True
            output_dims = output_dims[1]
        else:
            self.return_sequences = False
        self.add = tf.keras.layers.Add()
        self.transformer_blocks = []
        for i in range(n_layers):
            transformer_block = keras_transformer.TransformerBlock(
                n_heads, dropout, attn_dropout, activation='relu',
                use_masking=self.return_sequences, vanilla_wiring=True)
            self.transformer_blocks.append(transformer_block)
            setattr(self, f'transformer_{i}', transformer_block)
        self.to_segments = PaddedToSegments()
        self.aggregation = SegmentAggregation(
            aggregation_fn, cumulative=self.return_sequences)
        self.out_mlp = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(n_dims, activation='relu'),
                tf.keras.layers.Dense(output_dims, output_activation)
            ],
            name='out_mlp'
        )

    def build(self, input_shapes):
        demo, times, values, measurements, lengths = input_shapes
        self.positional_encoding.build(times)
        self.demo_embedding.build(demo)
        embedding_input = (
            None, values[-1] + measurements[-1])
        self.element_embedding.build(embedding_input)
        embedding_shape = self.element_embedding.compute_output_shape(embedding_input)
        self.add.build([embedding_shape, embedding_shape])
        for block in self.transformer_blocks:
            block.build(tuple(embedding_shape))
        self.to_segments.build(embedding_shape)
        segments = self.to_segments.compute_output_shape(embedding_shape)
        aggregated_output = (
            self.aggregation.compute_output_shape(segments))
        self.out_mlp.build(aggregated_output)
        self.built = True
        # super().build(input_shapes)

    def call(self, inputs):
        demo, times, values, measurements, lengths = inputs
        transformed_times = self.positional_encoding(times)
        value_modality_embedding = tf.concat(
            (
                values,
                tf.cast(measurements, tf.float32)
            ),
            axis=-1
        )

        # Somehow eager execution and graph mode behave differently.
        # In graph mode legths has an additional dimension
        if len(lengths.get_shape()) == 2:
            lengths = tf.squeeze(lengths, -1)
        mask = tf.sequence_mask(lengths+1, name='mask')

        demo_embedded = self.demo_embedding(demo)
        embedded = self.element_embedding(value_modality_embedding)
        combined = self.add([transformed_times, embedded])
        combined = tf.concat(
            [tf.expand_dims(demo_embedded, 1), combined], axis=1)
        transformer_out = combined
        for block in self.transformer_blocks:
            transformer_out = block(transformer_out, mask=mask)

        collected_values, segment_ids = self.to_segments(transformer_out, mask)

        aggregated_values = self.aggregation(collected_values, segment_ids)
        output = self.out_mlp(aggregated_values)

        if self.return_sequences:
            # If we should return sequences, then we need to transform the
            # output back into a tensor of the right shape
            valid_observations = tf.cast(tf.where(mask), tf.int32)
            output = tf.scatter_nd(
                valid_observations,
                output,
                tf.concat([tf.shape(mask), tf.shape(output)[-1:]], axis=0)
            )
            # Cut of the prediction only based on demographics
            output = output[:, 1:]
            # Indicate that the tensor contains invalid values
            output._keras_mask = mask[:, 1:]

        return output

    @classmethod
    def get_hyperparameters(cls):
        import tensorboard.plugins.hparams.api as hp
        from ..training_utils import HParamWithDefault
        return [
            HParamWithDefault(
                'n_dims',
                hp.Discrete([64, 128, 256, 512, 1024]),
                default=128
            ),
            HParamWithDefault(
                'n_heads',
                hp.Discrete([2, 4, 8]),
                default=4
            ),
            HParamWithDefault(
                'n_layers',
                hp.Discrete([1, 2, 3, 4, 5]),
                default=1
            ),
            HParamWithDefault(
                'dropout',
                hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4]),
                default=0.
            ),
            HParamWithDefault(
                'attn_dropout',
                hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4]),
                default=0.
            ),
            HParamWithDefault(
                'aggregation_fn',
                hp.Discrete(['mean', 'sum', 'max']),
                default='mean'
            ),
            HParamWithDefault(
                'max_timescale',
                hp.Discrete([10., 100., 1000.]),
                default=100.
            )
        ]

    @classmethod
    def from_hyperparameter_dict(cls, task, hparams):
        return cls(
            output_activation=task.output_activation,
            output_dims=task.n_outputs,
            n_dims=hparams['n_dims'],
            n_heads=hparams['n_heads'],
            n_layers=hparams['n_layers'],
            dropout=hparams['dropout'],
            attn_dropout=hparams['attn_dropout'],
            aggregation_fn=hparams['aggregation_fn'],
            max_timescale=hparams['max_timescale'],
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return self._config

    def data_preprocessing_fn(self):
        def add_time_dim(inputs, label):
            demo, times, values, measurements, lengths = inputs
            times = tf.expand_dims(times, -1)
            return (demo, times, values, measurements, lengths), label
        return add_time_dim
