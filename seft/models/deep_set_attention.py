from collections.abc import Sequence
from itertools import chain
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.python.framework.smart_cond import smart_cond

from .set_utils import (
    build_dense_dropout_model, PaddedToSegments, SegmentAggregation,
    cumulative_softmax_weighting, cumulative_segment_mean)
from .utils import segment_softmax


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


class CumulativeSetAttentionLayer(tf.keras.layers.Layer):
    dense_options = {
        'activation': 'relu',
        'kernel_initializer': 'he_uniform'
    }
    def __init__(self, n_layers=2, width=128, latent_width=128,
                 aggregation_function='mean',
                 dot_prod_dim=64, n_heads=4, attn_dropout=0.3):
        super().__init__()
        assert aggregation_function == 'mean'
        self.width = width
        self.dot_prod_dim = dot_prod_dim
        self.attn_dropout = attn_dropout
        self.n_heads = n_heads
        self.psi = build_dense_dropout_model(
            n_layers, width, 0., self.dense_options)
        self.psi.add(Dense(latent_width, **self.dense_options))
        self.rho = Dense(latent_width, **self.dense_options)

    def build(self, input_shape):
        self.psi.build(input_shape)
        encoded_shape = self.psi.compute_output_shape(input_shape)
        self.rho.build(encoded_shape)
        self.W_k = self.add_weight(
            'W_k',
            (encoded_shape[-1] + input_shape[-1], self.dot_prod_dim*self.n_heads),
            initializer='he_uniform'
        )
        self.W_q = self.add_weight(
            'W_q', (self.n_heads, self.dot_prod_dim),
            initializer=tf.keras.initializers.Zeros()
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_heads)

    def call(self, inputs, segment_ids, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        encoded = self.psi(inputs)

        # cumulative mean aggregation
        agg = cumulative_segment_mean(encoded, segment_ids)
        agg = self.rho(agg)

        combined = tf.concat([inputs, agg], axis=-1)
        keys = tf.matmul(combined, self.W_k)
        keys = tf.stack(tf.split(keys, self.n_heads, -1), 1)
        keys = tf.expand_dims(keys, axis=2)
        # should have shape (el, heads, 1, dot_prod_dim)
        queries = tf.expand_dims(tf.expand_dims(self.W_q, -1), 0)
        # should have shape (1, heads, dot_prod_dim, 1)
        preattn = tf.matmul(keys, queries) / tf.sqrt(float(self.dot_prod_dim))
        preattn = tf.squeeze(tf.squeeze(preattn, -1), -1)
        return preattn


class SetAttentionLayer(tf.keras.layers.Layer):
    dense_options = {
        'activation': 'relu',
        'kernel_initializer': 'he_uniform'
    }
    def __init__(self, n_layers=2, width=128, latent_width=128,
                 aggregation_function='mean',
                 dot_prod_dim=64, n_heads=4, attn_dropout=0.3):
        super().__init__()
        self.width = width
        self.dot_prod_dim = dot_prod_dim
        self.attn_dropout = attn_dropout
        self.n_heads = n_heads
        self.psi = build_dense_dropout_model(
            n_layers, width, 0., self.dense_options)
        self.psi.add(Dense(latent_width, **self.dense_options))
        self.psi_aggregation = SegmentAggregation(aggregation_function)
        self.rho = Dense(latent_width, **self.dense_options)

    def build(self, input_shape):
        self.psi.build(input_shape)
        encoded_shape = self.psi.compute_output_shape(input_shape)
        agg_shape = self.psi_aggregation.compute_output_shape(encoded_shape)
        self.rho.build(agg_shape)
        self.W_k = self.add_weight(
            'W_k',
            (encoded_shape[-1] + input_shape[-1], self.dot_prod_dim*self.n_heads),
            initializer='he_uniform'
        )
        self.W_q = self.add_weight(
            'W_q', (self.n_heads, self.dot_prod_dim),
            initializer=tf.keras.initializers.Zeros()
        )

    def call(self, inputs, segment_ids, lengths, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        def dropout_attn(input_tensor):
            if self.attn_dropout > 0:
                mask = (
                    tf.random.uniform(
                        tf.shape(input_tensor)[:-1]
                    ) < self.attn_dropout)
                return (
                    input_tensor
                    + tf.expand_dims(tf.cast(mask, tf.float32), -1) * -1e9
                )
            else:
                return tf.identity(input_tensor)

        encoded = self.psi(inputs)
        agg = self.psi_aggregation(encoded, segment_ids)
        agg = self.rho(agg)
        agg_scattered = tf.gather_nd(agg, tf.expand_dims(segment_ids, -1))
        combined = tf.concat([inputs, agg_scattered], axis=-1)
        keys = tf.matmul(combined, self.W_k)
        keys = tf.stack(tf.split(keys, self.n_heads, -1), 1)
        keys = tf.expand_dims(keys, axis=2)
        # should have shape (el, heads, 1, dot_prod_dim)
        queries = tf.expand_dims(tf.expand_dims(self.W_q, -1), 0)
        # should have shape (1, heads, dot_prod_dim, 1)
        preattn = tf.matmul(keys, queries) / tf.sqrt(float(self.dot_prod_dim))
        preattn = tf.squeeze(preattn, -1)
        preattn = smart_cond(
            training,
            lambda: dropout_attn(preattn),
            lambda: tf.identity(preattn)
        )

        per_head_preattn = tf.unstack(preattn, axis=1)
        attentions = []
        for pre_attn in per_head_preattn:
            attentions.append(segment_softmax(pre_attn, segment_ids))
        return attentions

    def compute_output_shape(self, input_shape):
        return list(chain(input_shape[:-1], (self.n_heads, )))


class IdentityLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def compute_output_shape(self, input_shapes):
        return input_shapes

    def call(self, inputs, **kwargs):
        return inputs


class DeepSetAttentionModel(tf.keras.Model):
    dense_options = {
        'activation': 'relu',
        'kernel_initializer': 'he_uniform'
    }

    def __init__(self, output_activation, output_dims, n_phi_layers, phi_width,
                 n_psi_layers, psi_width, psi_latent_width, dot_prod_dim,
                 n_heads, attn_dropout, latent_width, phi_dropout,
                 n_rho_layers, rho_width, rho_dropout, max_timescale,
                 n_positional_dims):
        self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }
        super().__init__()
        self.phi_width = phi_width
        self.to_segments = PaddedToSegments()
        # If we set n_positional_dims to 0, skip the positional encoding
        self.positional_encoding = (
            PositionalEncoding(max_timescale, n_positional_dims)
            if n_positional_dims != 0
            else IdentityLayer()
        )
        # We need the input dimensionality in order to determine the size of
        # the embedding for the demographics.
        self.demo_encoder = None
        if isinstance(output_dims, Sequence):
            # We have an online prediction scenario
            assert output_dims[0] is None
            self.return_sequences = True
            output_dims = output_dims[1]
        else:
            self.return_sequences = False

        # Build phi architecture
        self.phi = build_dense_dropout_model(
            n_phi_layers, phi_width, phi_dropout, self.dense_options)
        self.phi.add(Dense(latent_width, **self.dense_options))
        self.latent_width = latent_width
        self.n_heads = n_heads

        if self.return_sequences:
            self.attention = CumulativeSetAttentionLayer(
                n_psi_layers, psi_width, psi_latent_width,
                dot_prod_dim=dot_prod_dim, n_heads=n_heads,
                attn_dropout=attn_dropout
            )
        else:
            self.attention = SetAttentionLayer(
                n_psi_layers, psi_width, psi_latent_width,
                dot_prod_dim=dot_prod_dim, n_heads=n_heads,
                attn_dropout=attn_dropout
            )

        self.aggregation = SegmentAggregation(
            aggregation_fn='sum',
            cumulative=self.return_sequences
        )

        # Build rho architecture
        self.rho = build_dense_dropout_model(
            n_rho_layers, rho_width, rho_dropout, self.dense_options)
        self.rho.add(Dense(output_dims, activation=output_activation))
        self._n_modalities = None

    def build(self, input_shapes):
        if self.return_sequences:
            demo, times, values, measurements, lengths, inverse_timepoints, pred_lengths = input_shapes
        else:
            demo, times, values, measurements, lengths = input_shapes
        self.positional_encoding.build(times)
        transformed_times = (
            self.positional_encoding.compute_output_shape(times))
        mod_shape = self._n_modalities
        phi_input_dim = transformed_times[-1] + values[-1] + mod_shape
        self.demo_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.phi_width, activation='relu'),
                tf.keras.layers.Dense(phi_input_dim)
            ],
            name='demo_encoder'
        )
        self.demo_encoder.build(demo)
        if self.return_sequences:
            phi_input = (None, phi_input_dim)
            self.phi.build(phi_input)
            phi_output = self.phi.compute_output_shape(phi_input)
            self.attention.build(phi_input)
            attention_output = self.attention.compute_output_shape(phi_input)
            aggregated_output = [
                phi_output[0], phi_output[1] * attention_output[1]]
            self.rho.build(aggregated_output)
        else:
            phi_input = (None, phi_input_dim)
            self.phi.build(phi_input)
            phi_output = self.phi.compute_output_shape(phi_input)
            self.attention.build(phi_input)
            attention_output = self.attention.compute_output_shape(phi_input)
            aggregated_output = self.aggregation.compute_output_shape(
                [phi_output[0], phi_output[1] * attention_output[1]])
            self.rho.build(aggregated_output)

    def call(self, inputs):
        if self.return_sequences:
            demo, times, values, measurements, lengths, elem_per_tp, pred_lengths = inputs
            if len(pred_lengths.get_shape()) == 2:
                pred_lengths = tf.squeeze(pred_lengths, -1)
        else:
            demo, times, values, measurements, lengths = inputs
        transformed_times = self.positional_encoding(times)

        # Transform modalities
        transformed_measurements = tf.one_hot(
            measurements, self._n_modalities, dtype=tf.float32)

        combined_values = tf.concat(
            (
                transformed_times,
                values,
                transformed_measurements
            ),
            axis=-1
        )
        demo_encoded = self.demo_encoder(demo)
        combined_with_demo = tf.concat(
            [tf.expand_dims(demo_encoded, 1), combined_values], axis=1)

        # Somehow eager execution and graph mode behave differently.
        # In graph mode lengths has an additional dimension
        if len(lengths.get_shape()) == 2:
            lengths = tf.squeeze(lengths, -1)

        if self.return_sequences:
            # We additionally have the encoded demographics as a set element
            mask = tf.sequence_mask(lengths+1, name='mask')

            collected_values, segment_ids = self.to_segments(
                combined_with_demo, mask)

            preattentions = self.attention(collected_values, segment_ids)
            encoded = self.phi(collected_values)
            agg = cumulative_softmax_weighting(
                encoded, preattentions, segment_ids)
            # Remove heads dimension
            agg = tf.reshape(
                agg,
                tf.stack([tf.shape(agg)[0], tf.constant(-1)], axis=0)
            )

            predictions_mask = tf.sequence_mask(pred_lengths)
            gathered_time_indices, batch_indices = self.to_segments(
                elem_per_tp, predictions_mask)

            # Compute index of the last observation associated with the
            # provided time.
            prediction_indices = tf.math.cumsum(gathered_time_indices)
            # Add an offset for each instance to account for demographics. This
            # offset decreases for each later index in the batch. Thus we can
            # use the batch indices.
            prediction_indices += batch_indices

            gathered_embeddings = tf.gather_nd(
                agg, prediction_indices[:, None])
            # Lost shape information
            gathered_embeddings.set_shape([None, None])
            output = self.rho(gathered_embeddings)

            valid_predictions = tf.cast(tf.where(predictions_mask), tf.int32)

            output = tf.scatter_nd(
                valid_predictions,
                output,
                tf.concat(
                    [tf.shape(predictions_mask), tf.shape(output)[-1:]],
                    axis=0
                )
            )
            # tf.print(tf.shape(output), tf.shape(mask))
            output._keras_mask = predictions_mask
            return output
        else:
            # We additionally have the encoded demographics as a set element
            mask = tf.sequence_mask(lengths+1, name='mask')

            collected_values, segment_ids = self.to_segments(
                combined_with_demo, mask)

            encoded = self.phi(collected_values)
            attentions = self.attention(collected_values, segment_ids, lengths)

            weighted_values = []
            for attention in attentions:
                weighted_values.append(encoded * attention)

            aggregated_values = self.aggregation(
                tf.concat(weighted_values, axis=-1), segment_ids)
            return self.rho(aggregated_values)

    def get_attentions(self, inputs):
        demo, times, values, measurements, lengths = inputs
        transformed_times = self.positional_encoding(times)

        # Transform modalities
        if self._n_modalities > 100:
            # Use an embedding instead of one hot encoding when we have a very
            # high number of modalities
            transformed_measurements = self.modality_embedding(measurements)
        else:
            transformed_measurements = tf.one_hot(
                measurements, self._n_modalities, dtype=tf.float32)

        combined_values = tf.concat(
            (
                transformed_times,
                values,
                transformed_measurements
            ),
            axis=-1
        )
        demo_encoded = self.demo_encoder(demo)
        combined_with_demo = tf.concat(
            [tf.expand_dims(demo_encoded, 1), combined_values], axis=1)
        # Somehow eager execution and graph mode behave differently.
        # In graph mode legths has an additional dimension
        if len(lengths.get_shape()) == 2:
            lengths = tf.squeeze(lengths, -1)

        # We additionally have the encoded demographics as a set element
        mask = tf.sequence_mask(lengths+1, name='mask')
        valid_observations = tf.cast(tf.where(mask), tf.int32)
        out_shape = tf.concat(
            [
                tf.shape(combined_with_demo)[:-1],
                tf.constant([1])
            ],
            axis=0,
        )

        collected_values, segment_ids = self.to_segments(combined_with_demo, mask)

        attentions = self.attention(collected_values, segment_ids, lengths)

        demo_attentions = []
        ts_attentions = []

        for attention in attentions:
            dist_attention = tf.scatter_nd(
                    valid_observations, attention, out_shape)
            demo_attentions.append(dist_attention[:, 0])
            ts_attentions.append(dist_attention[:, 1:])
        return demo_attentions, ts_attentions

    def _evtl_create_embedding_layer(self):
        if self._n_modalities > 100 and not hasattr(self, 'modality_embedding'):
            self.modality_embedding = tf.keras.layers.Embedding(
                self._n_modalities, 64)

    @classmethod
    def get_hyperparameters(cls):
        import tensorboard.plugins.hparams.api as hp
        from ..training_utils import HParamWithDefault
        return [
            HParamWithDefault(
                'n_phi_layers', hp.Discrete([1, 2, 3, 4, 5]), default=3),
            HParamWithDefault(
                'phi_width',
                hp.Discrete([16, 32, 64, 128, 256, 512]),
                default=32
            ),
            HParamWithDefault(
                'phi_dropout',
                hp.Discrete([0.0, 0.1, 0.2, 0.3]),
                default=0.
            ),
            HParamWithDefault(
                'n_psi_layers',
                hp.Discrete([2]),
                default=2
            ),
            HParamWithDefault(
                'psi_width',
                hp.Discrete([64]),
                default=64
            ),
            HParamWithDefault(
                'psi_latent_width',
                hp.Discrete([128]),
                default=128
            ),
            HParamWithDefault(
                'dot_prod_dim',
                hp.Discrete([128]),
                default=128
            ),
            HParamWithDefault(
                'n_heads',
                hp.Discrete([4]),
                default=4
            ),
            HParamWithDefault(
                'attn_dropout',
                hp.Discrete([0.0, 0.1, 0.25, 0.5]),
                default=0.1
            ),
            HParamWithDefault(
                'latent_width',
                hp.Discrete([32, 64, 128, 256, 512, 1024, 2048]),
                default=128
            ),
            HParamWithDefault(
                'n_rho_layers', hp.Discrete([1, 2, 3, 4, 5]), default=3),
            HParamWithDefault(
                'rho_width',
                hp.Discrete([16, 32, 64, 128, 256, 512]),
                default=32
            ),
            HParamWithDefault(
                'rho_dropout',
                hp.Discrete([0.0, 0.1, 0.2, 0.3]),
                default=0.
            ),
            HParamWithDefault(
                'max_timescale',
                hp.Discrete([10., 100., 1000.]),
                default=100.
            ),
            HParamWithDefault(
                'n_positional_dims',
                hp.Discrete([4, 8, 16]),
                default=4
            )
        ]

    @classmethod
    def from_hyperparameter_dict(cls, task, hparams):
        return cls(
            output_activation=task.output_activation,
            output_dims=task.n_outputs,
            n_phi_layers=hparams['n_phi_layers'],
            phi_width=hparams['phi_width'],
            n_psi_layers=hparams['n_psi_layers'],
            psi_width=hparams['psi_width'],
            psi_latent_width=hparams['psi_latent_width'],
            dot_prod_dim=hparams['dot_prod_dim'],
            n_heads=hparams['n_heads'],
            attn_dropout=hparams['attn_dropout'],
            latent_width=hparams['latent_width'],
            phi_dropout=hparams['phi_dropout'],
            n_rho_layers=hparams['n_rho_layers'],
            rho_width=hparams['rho_width'],
            rho_dropout=hparams['rho_dropout'],
            max_timescale=hparams['max_timescale'],
            n_positional_dims=hparams['n_positional_dims']
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return self._config

    def data_preprocessing_fn(self):
        def flatten_unaligned_measurements(ts, labels):
            # Ignore demographics for now
            demo, X, Y, measurements, lengths = ts
            if self._n_modalities is None:
                self._n_modalities = int(measurements.get_shape()[-1])
            X = tf.expand_dims(X, -1)
            measurement_positions = tf.cast(tf.where(measurements), tf.int32)
            X_indices = measurement_positions[:, 0]
            Y_indices = measurement_positions[:, 1]

            gathered_X = tf.gather(X, X_indices)
            gathered_Y = tf.gather_nd(Y, measurement_positions)
            gathered_Y = tf.expand_dims(gathered_Y, axis=-1)

            length = tf.shape(X_indices)[0]
            if self.return_sequences:
                # We need to know now many prediction values each instance
                # should have when doing online prediction
                prediction_length = tf.shape(labels)[0]
                counts = tf.reduce_sum(tf.cast(measurements, tf.int64), axis=1)
                return (demo, gathered_X, gathered_Y, Y_indices, length, counts, prediction_length), labels
            else:
                return (demo, gathered_X, gathered_Y, Y_indices, length), labels

        return flatten_unaligned_measurements

    @classmethod
    def get_default(cls, task):
        hyperparams = cls.get_hyperparameters()
        return cls.from_hyperparameter_dict(
            task,
            {
                h.name: h._default for h in hyperparams
            }
        )


class DeepSetAttentionNoPosModel(DeepSetAttentionModel):
    def __init__(self, output_activation, output_dims, **kwargs):
        super().__init__(output_activation, output_dims, **kwargs,
                         max_timescale=0,
                         n_positional_dims=0)

    @classmethod
    def get_hyperparameters(cls):
        parent_hyperparameters = super().get_hyperparameters()
        return [
            hp for hp in parent_hyperparameters
            if hp.name not in ['max_timescale', 'n_positional_dims']
        ]

    @classmethod
    def from_hyperparameter_dict(cls, task, hparams):
        return cls(
            output_activation=task.output_activation,
            output_dims=task.n_outputs,
            n_phi_layers=hparams['n_phi_layers'],
            phi_width=hparams['phi_width'],
            n_psi_layers=hparams['n_psi_layers'],
            psi_width=hparams['psi_width'],
            psi_latent_width=hparams['psi_latent_width'],
            dot_prod_dim=hparams['dot_prod_dim'],
            n_heads=hparams['n_heads'],
            attn_dropout=hparams['attn_dropout'],
            latent_width=hparams['latent_width'],
            phi_dropout=hparams['phi_dropout'],
            n_rho_layers=hparams['n_rho_layers'],
            rho_width=hparams['rho_width'],
            rho_dropout=hparams['rho_dropout'],
        )
