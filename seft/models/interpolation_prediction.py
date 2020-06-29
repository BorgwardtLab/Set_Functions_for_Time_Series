"""Interpolation-Prediction Networks."""
from collections.abc import Sequence

import numpy as np
import tensorflow as tf

K = tf.keras.backend


class single_channel_interp(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.reconstruction = False
        super(single_channel_interp, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape [batch, features, time_stamp]
        # For a first ignore the reconstruction part of the loss, thus unly
        # divide by 3
        # self.d_dim = input_shape[1]/4
        assert int(input_shape[1]) % 3 == 0
        self.d_dim = input_shape[1] // 3
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self.d_dim, ),
            initializer=tf.keras.initializers.Constant(value=0.0),
            trainable=True)
        super(single_channel_interp, self).build(input_shape)

    def call(self, x, interpolation_grid, reconstruction=False):
        self.reconstruction = reconstruction
        x_t = x[:, :self.d_dim, :]
        m = x[:, self.d_dim:2*self.d_dim, :]
        d = x[:, 2*self.d_dim:3*self.d_dim, :]
        time_stamp = tf.shape(x)[2]
        if reconstruction:
            ref_t = K.tile(d[:, :, None, :], tf.stack((1, 1, time_stamp, 1)))
            output_dim = time_stamp
        else:
            ref_t = tf.expand_dims(
                tf.expand_dims(interpolation_grid, 1), 1)
            output_dim = tf.shape(interpolation_grid)[-1]
        d = K.tile(d[:, :, :, None], (1, 1, 1, output_dim))
        mask = K.tile(m[:, :, :, None], (1, 1, 1, output_dim))
        x_t = K.tile(x_t[:, :, :, None], (1, 1, 1, output_dim))
        norm = (d - ref_t)*(d - ref_t)
        a = tf.ones(tf.stack((self.d_dim, time_stamp, output_dim), axis=0))
        pos_kernel = K.log(1 + K.exp(self.kernel))
        alpha = a*pos_kernel[:, np.newaxis, np.newaxis]
        w = tf.reduce_logsumexp(-alpha*norm + K.log(mask), axis=2)
        w1 = K.tile(w[:, :, None, :], (1, 1, time_stamp, 1))
        w1 = K.exp(-alpha*norm + K.log(mask) - w1)
        y = K.sum(w1*x_t, axis=2)
        if reconstruction:
            rep1 = tf.concat([y, w], 1)
        else:
            w_t = tf.reduce_logsumexp(-10.0*alpha*norm + K.log(mask), axis=2)
            # kappa = 10
            w_t = K.tile(w_t[:, :, None, :], (1, 1, time_stamp, 1))
            w_t = K.exp(-10.0*alpha*norm + K.log(mask) - w_t)
            y_trans = K.sum(w_t*x_t, axis=2)
            rep1 = tf.concat([y, w, y_trans], 1)
        return rep1

    def compute_output_shape(self, input_shape):
        if self.reconstruction:
            return (input_shape[0], 2*self.d_dim, None)
        return (input_shape[0], 3*self.d_dim, None)


class cross_channel_interp(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        self.reconstruction = False
        super(cross_channel_interp, self).__init__(**kwargs)

    def build(self, input_shape):
        self.d_dim = int(input_shape[1] // 3)
        self.cross_channel_interp = self.add_weight(
            name='cross_channel_interp',
            shape=(self.d_dim, self.d_dim),
            initializer=tf.keras.initializers.Identity(gain=1.0),
            trainable=True)

        super(cross_channel_interp, self).build(input_shape)

    def call(self, x, reconstruction=False):
        self.reconstruction = reconstruction
        self.output_dim = tf.shape(x)[-1]
        cross_channel_interp = self.cross_channel_interp
        y = x[:, :self.d_dim, :]
        w = x[:, self.d_dim:2*self.d_dim, :]
        intensity = K.exp(w)
        y = tf.transpose(y, perm=[0, 2, 1])
        w = tf.transpose(w, perm=[0, 2, 1])
        w2 = w
        w = K.tile(w[:, :, :, None], (1, 1, 1, self.d_dim))
        den = tf.reduce_logsumexp(w, axis=2)
        w = K.exp(w2 - den)
        mean = K.mean(y, axis=1)
        mean = K.tile(mean[:, None, :], tf.stack((1, self.output_dim, 1)))
        w2 = K.dot(w*(y - mean), cross_channel_interp) + mean
        rep1 = tf.transpose(w2, perm=[0, 2, 1])
        if reconstruction is False:
            y_trans = x[:, 2*self.d_dim:3*self.d_dim, :]
            y_trans = y_trans - rep1  # subtracting smooth from transient part
            rep1 = tf.concat([rep1, intensity, y_trans], 1)
        return rep1

    def compute_output_shape(self, input_shape):
        if self.reconstruction:
            return (input_shape[0], self.d_dim, input_shape[2])
        return (input_shape[0], 3*self.d_dim, input_shape[2])


class InterpolationPredictionModel(tf.keras.Model):
    def __init__(self, output_activation, output_dims, n_units,
                 imputation_stepsize, dropout, recurrent_dropout,
                 reconst_fraction):
        self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }
        super().__init__()

        self.imputation_stepsize = imputation_stepsize
        self.reconst_fraction = reconst_fraction
        self.eps = 1e-9
        self.singe_channel_interp = single_channel_interp()
        self.cross_channel_interp = cross_channel_interp()
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
            n_units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            return_sequences=self.return_sequences
        )
        self.output_layer = tf.keras.layers.Dense(
            output_dims, activation=output_activation)

    def build(self, input_shapes):
        # Layers expect (bs, feature, tp) shape
        demo_shape, times_shape, values_shape, measurements_shape = input_shapes[:4]
        # Layers expect the inputs to be concatenated along the axis 1.
        concat_feature_dim = (
            times_shape[1] + values_shape[1] + measurements_shape[1])
        sic_input = (values_shape[0], concat_feature_dim, values_shape[2])
        self.singe_channel_interp.build(sic_input)
        sic_output = self.singe_channel_interp.compute_output_shape(sic_input)

        self.cross_channel_interp.build(sic_output)
        crc_output = \
            self.cross_channel_interp.compute_output_shape(sic_output)

        self.demo_encoder.build(demo_shape)
        # We transpose back before calling the rnn, such that (bs, tp, feature)
        rnn_input = (crc_output[0], crc_output[2], crc_output[1])
        self.rnn.build(rnn_input)
        rnn_output = self.rnn.compute_output_shape(rnn_input)
        self.output_layer.build(rnn_output)
        super().build(input_shapes)

    def data_preprocessing_fn(self):
        def prepro_fn(ts, label):
            demo, X, Y, measurements, lengths = ts
            X = tf.expand_dims(X, -1)
            # Check if a value was never measured. If this is the case, add an
            # observation at timepoint t=0 with the mean. We assume mean
            # centered data, thus the mean is zero.
            n_observed_values = tf.reduce_sum(
                tf.cast(tf.equal(measurements, False), tf.int32),
                axis=0,
            )
            nothing_ever_observed = tf.squeeze(
                tf.where(tf.equal(n_observed_values, lengths)),
                axis=-1
            )
            indices = tf.stack(
                [tf.zeros_like(nothing_ever_observed), nothing_ever_observed],
                axis=1
            )
            Y = tf.tensor_scatter_nd_update(
                Y, indices, tf.zeros(tf.shape(indices)[0]))
            measurements = tf.tensor_scatter_nd_update(
                measurements,
                indices,
                tf.ones(tf.shape(indices)[0], dtype=bool)
            )

            if self.return_sequences:
                # In the online scenario use the timepoints provided as a grid
                grid = X[:, 0]
                grid_length = tf.cast(tf.shape(grid)[0], tf.int32)
            else:
                # Generate a grid where imputation should take place
                end_time = tf.reduce_max(X)
                grid = tf.range(
                    end_time + self.imputation_stepsize,
                    delta=self.imputation_stepsize
                )
                grid_length = tf.cast(tf.shape(grid)[0], tf.int32)

            # We require the timepoints to be of same dimensionality than the
            # measurement values.
            X = tf.tile(X, [1, Y.get_shape()[-1]])
            # Further the layers expect (bs, feature, tp inputs), so we need to
            # transpose
            return (
                (
                    demo,
                    tf.transpose(X),
                    tf.transpose(Y),
                    tf.transpose(measurements),
                    tf.transpose(grid),
                    grid_length,
                    lengths
                ),
                label
            )
        return prepro_fn

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        demo, times, values, measurements, grid, grid_lengths, lengths = inputs

        # Compute classification
        mask = tf.sequence_mask(
            tf.squeeze(grid_lengths, axis=-1),
            tf.shape(grid)[-1], dtype=tf.int32)
        layer_input = tf.concat(
            (values, tf.cast(measurements, tf.float32), times), axis=1)
        sic_output = self.singe_channel_interp(layer_input, grid)
        crc_output = self.cross_channel_interp(sic_output)
        rnn_input = tf.transpose(crc_output, perm=[0, 2, 1])

        demo_encoded = self.demo_encoder(demo)
        rnn_output = self.rnn(rnn_input, mask=mask, initial_state=demo_encoded)

        # Compute reconstruction loss
        # Reconst mask is one if values are included
        reconst_mask = tf.greater(
            tf.random.uniform(tf.shape(measurements)), self.reconst_fraction)
        context_measurements = tf.logical_and(measurements, reconst_mask)

        # It could be that a feature does not contain a single observation,
        # in this case add a observation on the first timepoint. As the data is
        # mean centered an non observed values are simply represented by 0, we
        # dont need to fill in values into the values tensor.
        nothing_observed = tf.equal(
            tf.reduce_sum(
                tf.cast(context_measurements, tf.int32), -1, keepdims=True),
            0
        )
        n_elements = tf.reduce_sum(tf.cast(nothing_observed, tf.int32))
        context_measurements = tf.tensor_scatter_nd_update(
            context_measurements,
            tf.where(nothing_observed),
            tf.ones(n_elements, dtype=bool)
        )

        reconst_input = tf.concat(
            (values, tf.cast(context_measurements, tf.float32), times), axis=1)
        sic_reconst = self.singe_channel_interp(
            reconst_input, None, reconstruction=True)
        crc_reconst = self.cross_channel_interp(
            sic_reconst, reconstruction=True)

        # Score reconstruction on points which are not available as context
        target_measurements = tf.logical_and(
            measurements,
            tf.logical_not(context_measurements)
        )
        target_measurements = tf.cast(target_measurements, tf.float32)
        squared_error = target_measurements * (values - crc_reconst) ** 2
        # It can happen, that we have very little data availabele then the
        # computing the reconstruction error is almost impossible as no data is
        # selected to be excluded for reconstruction. This leads to a division
        # by zero and to nans. For now fix this by adding eps to the division.
        instance_wise_reconst_error = (
            tf.reduce_sum(squared_error, axis=[1, 2]) /
            (tf.reduce_sum(target_measurements, axis=[1, 2]) + self.eps)
        )
        reconst_error = tf.keras.metrics.Mean(name='reconst_error')(
            instance_wise_reconst_error)

        # Only regard reconstruction component of loss during fitting, ignore
        # for validation loss (in order to allow model selection based on
        # classification performance)
        def get_mean_reconst_loss():
            return tf.reduce_mean(instance_wise_reconst_error)
        def no_loss():
            return tf.zeros((), dtype=tf.float32)

        reconstruction_loss = tf.cond(
            training,
            get_mean_reconst_loss,
            no_loss
        )
        self.add_loss(reconstruction_loss, inputs=True)
        self.add_metric(reconst_error)

        return self.output_layer(rnn_output)

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
            ),
            HParamWithDefault(
                'imputation_stepsize',
                hp.Discrete([0.5, 1., 2.5, 5.]),
                default=1.
            ),
            HParamWithDefault(
                'reconst_fraction',
                hp.Discrete([0.05, 0.1, 0.2, 0.5, 0.75]),
                default=0.05
            )
        ]

    @classmethod
    def from_hyperparameter_dict(cls, task, hparams):
        return cls(
            output_activation=task.output_activation,
            output_dims=task.n_outputs,
            n_units=hparams['n_units'],
            dropout=hparams['dropout'],
            recurrent_dropout=hparams['recurrent_dropout'],
            imputation_stepsize=hparams['imputation_stepsize'],
            reconst_fraction=hparams['reconst_fraction']
    )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return self._config
