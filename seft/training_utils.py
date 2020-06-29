"""Utility functions for training and evaluation."""
import math
import random

from collections.abc import Sequence

import tensorflow as tf
import tensorflow_datasets as tfds
import medical_ts_datasets
import seft.models
from tensorflow.data.experimental import AUTOTUNE
import tensorboard.plugins.hparams.api as hp
from tensorboard.plugins.hparams import api_pb2

get_output_shapes = tf.compat.v1.data.get_output_shapes
get_output_types = tf.compat.v1.data.get_output_types
make_one_shot_iterator = tf.compat.v1.data.make_one_shot_iterator


def build_hyperparameter_metrics(evaluation_metrics):
    metrics = []
    # Add training metrics
    metrics.append(hp.Metric(
        'batch_loss', display_name='loss (train)'))
    metrics.append(hp.Metric(
        'batch_acc', display_name='accuracy (train)'))

    # Add validation metrics
    metrics.append(hp.Metric(
        'epoch_val_loss', display_name='loss (validation)'))
    metrics.append(hp.Metric(
        'epoch_val_acc', display_name='accuracy (validation)'))
    metrics.append(hp.Metric(
        'best_val_loss', display_name='best loss (validation)'))
    metrics.append(hp.Metric(
        'best_val_acc', display_name='best accuracy (validation)'))

    # Add evaluation metrics
    metrics.extend(
        [
            hp.Metric(
                'epoch_val_' + metric, display_name=metric + ' (validation)')
            for metric in evaluation_metrics
        ]
    )
    return metrics


def init_hyperparam_space(logdir, hparams, metrics):
    # Add dataset and model as hyperparameters
    hparams = [
        hp.HParam('dataset', hp.Discrete(medical_ts_datasets.builders)),
        hp.HParam('model', hp.Discrete(seft.models.__all__))
    ] + list(hparams)
    sess = tf.compat.v1.keras.backend.get_session()
    with tf.compat.v2.summary.create_file_writer(logdir).as_default() as w:
        sess.run(w.init())
        sess.run(hp.hparams_config(hparams=hparams, metrics=metrics))
        sess.run(w.flush())


def get_padding_values(input_dataset_types, label_padding=-100):
    """Get a tensor of padding values fitting input_dataset_types.

    Here we pad everything with 0. and the labels with `label_padding`. This
    allows us to be able to recognize them later during the evaluation, even
    when the values have already been padded into batches.

    Args:
        tensor_shapes: Nested structure of tensor shapes.

    Returns:
        Nested structure of padding values where all are 0 except teh one
        corresponding to tensor_shapes[1], which is padded according to the
        `label_padding` value.

    """
    def map_to_zero(dtypes):
        if isinstance(dtypes, Sequence):
            return tuple((map_to_zero(d) for d in dtypes))
        return tf.cast(0., dtypes)

    def map_to_label_padding(dtypes):
        if isinstance(dtypes, Sequence):
            return tuple((map_to_zero(d) for d in dtypes))
        return tf.cast(label_padding, dtypes)

    if len(input_dataset_types) == 2:
        data_type, label_type = input_dataset_types
        return (
            map_to_zero(data_type),
            map_to_label_padding(label_type)
        )

    if len(input_dataset_types) == 3:
        data_type, label_type, sample_weight_type = input_dataset_types
        return (
            map_to_zero(data_type),
            map_to_label_padding(label_type),
            map_to_zero(sample_weight_type)
        )


def positive_instances(*args):
    if len(args) == 2:
        data, label = args
    if len(args) == 3:
        data, label, sample_weights = args

    return tf.math.equal(tf.reduce_max(label), 1)

def negative_instances(*args):
    if len(args) == 2:
        data, label = args
    if len(args) == 3:
        data, label, sample_weights = args
    return tf.math.equal(tf.reduce_max(label), 0)


def build_training_iterator(dataset_name, epochs, batch_size, prepro_fn,
                            balance=False, class_balance=None):
    dataset, dataset_info = tfds.load(
        dataset_name,
        split=tfds.Split.TRAIN,
        as_supervised=True,
        with_info=True
    )
    n_samples = dataset_info.splits['train'].num_examples
    steps_per_epoch = int(math.floor(n_samples / batch_size))
    if prepro_fn is not None:
        dataset = dataset.map(prepro_fn, num_parallel_calls=AUTOTUNE)

    if balance:
        majority_class = max(
            range(len(class_balance)), key=lambda i: class_balance[i])
        minority_class = min(
            range(len(class_balance)), key=lambda i: class_balance[i])

        n_majority = class_balance[majority_class] * n_samples
        n_minority = class_balance[minority_class] * n_samples
        # Generate two separate datasets using filter
        pos_data = (dataset
                    .filter(positive_instances)
                    .shuffle(
                        int(class_balance[1] * n_samples),
                        reshuffle_each_iteration=True)
                    .repeat()
                    )
        neg_data = (dataset
                    .filter(negative_instances)
                    .shuffle(
                        int(class_balance[0] * n_samples),
                        reshuffle_each_iteration=True)
                    .repeat()
                    )
        # And sample from them
        dataset = tf.data.experimental.sample_from_datasets(
            [pos_data, neg_data], weights=[0.5, 0.5])
        # One epoch should at least contain all negative examples or max
        # each instance of the minority class 3 times
        steps_per_epoch = min(
            math.ceil(2 * n_majority / batch_size),
            math.ceil(3 * 2 * n_minority / batch_size)
        )
    else:
        # Shuffle repeat and batch
        dataset = dataset.shuffle(n_samples, reshuffle_each_iteration=True)
        dataset = dataset.repeat(epochs)

    batched_dataset = dataset.padded_batch(
        batch_size,
        get_output_shapes(dataset),
        padding_values=get_padding_values(get_output_types(dataset)),
        drop_remainder=True
    )
    return batched_dataset.prefetch(AUTOTUNE), steps_per_epoch


def build_validation_iterator(dataset_name, batch_size, prepro_fn):
    """Build a validation iterator for a tensorflow datasets dataset.

    Args:
        dataset_name: Name of the tensoflow datasets dataset. To be used with
            tfds.load().
        epochs: Number of epochs to run
        batch_size: Batch size
        prepro_fn: Optional preprocessing function that should be applied to
            prior to batching.

    Returns:
        A tensorflow dataset which iterates through the validation dataset
           epoch times.

    """
    dataset, dataset_info = tfds.load(
        dataset_name,
        split=tfds.Split.VALIDATION,
        as_supervised=True,
        with_info=True
    )
    n_samples = dataset_info.splits['validation'].num_examples
    steps_per_epoch = int(math.ceil(n_samples / batch_size))
    if prepro_fn is not None:
        dataset = dataset.map(prepro_fn, num_parallel_calls=AUTOTUNE)

    # Batch
    batched_dataset = dataset.padded_batch(
        batch_size,
        get_output_shapes(dataset),
        padding_values=get_padding_values(dataset.output_types),
        drop_remainder=False
    )
    return batched_dataset, steps_per_epoch


def build_test_iterator(dataset_name, batch_size, prepro_fn):
    dataset, dataset_info = tfds.load(
        dataset_name,
        split=tfds.Split.TEST,
        as_supervised=True,
        with_info=True
    )
    n_samples = dataset_info.splits['test'].num_examples
    steps = int(math.floor(n_samples / batch_size))
    if prepro_fn is not None:
        dataset = dataset.map(prepro_fn, num_parallel_calls=AUTOTUNE)

    # Batch
    batched_dataset = dataset.padded_batch(
        batch_size,
        get_output_shapes(dataset),
        padding_values=get_padding_values(dataset.output_types),
        drop_remainder=False
    )
    return batched_dataset, steps


class HParamWithDefault(hp.HParam):
    """Subclass of tensorboard HParam, with additional default parameter."""

    def __init__(self, name, domain=None, display_name=None, description=None,
                 default=None):
        super().__init__(name, domain, display_name, description)
        self._default = default

    @property
    def default(self):
        return self._default


class LogRealInterval(hp.RealInterval):
    """Domain of real values on the log scale."""

    def __repr__(self):
        return "LogRealInterval(%r, %r)" % (self._min_value, self._max_value)

    def sample_uniform(self, rng=random):
        pre_exp = rng.uniform(
            math.log10(self._min_value), math.log10(self._max_value))
        return round(10 ** pre_exp, 5)
