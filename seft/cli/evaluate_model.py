"""Load a model and evaluate it on the metrics of the dataset."""
import argparse
import json

import seft.cli.silence_warnings
import tensorflow as tf
from tensorflow.data.experimental import Reducer
import tensorflow_datasets as tfds
import medical_ts_datasets
import numpy as np
from tqdm import tqdm

from seft.normalization import Normalizer
import seft.models
from seft.tasks import DATASET_TO_TASK_MAPPING
from seft.training_utils import build_validation_iterator, get_output_shapes, get_padding_values, get_output_types


def load_model(dataset_name, model_name, config_path, weights_path):
    model_class = getattr(seft.models, model_name)
    with open(config_path, 'r') as f:
        config = json.load(f)

    task = DATASET_TO_TASK_MAPPING[dataset_name]
    normalizer = Normalizer(dataset_name)
    model = model_class.from_hyperparameter_dict(task, config)
    model.compile(optimizer='adam', loss=task.loss)

    def combined_fn(ts, labels):
        normalized_ts, labels = \
            normalizer.get_normalization_fn()(ts, labels)
        prepro_fn = model.data_preprocessing_fn()
        if prepro_fn is None:
            return normalized_ts, labels
        return prepro_fn(normalized_ts, labels)

    # Fit model very briefly to initialize weights
    input_dataset, n_steps = build_validation_iterator(dataset_name, 16, combined_fn)
    model.fit(input_dataset, steps_per_epoch=1, verbose=0)
    model.load_weights(weights_path)
    return model


def expand_time_into_batch(index, inputs, labels):
    """Convert a single instance into a batch where each instance contains
    information up to timepoint.
    """
    demo, times, values, measurements, lengths = inputs
    demo_expanded = tf.tile(demo[None, :], [lengths, 1])
    times_expanded = tf.TensorArray(times.dtype, lengths)
    values_expanded = tf.TensorArray(values.dtype, lengths)
    measurements_expanded = tf.TensorArray(measurements.dtype, lengths)
    lengths_expanded = tf.TensorArray(lengths.dtype, lengths)
    labels_expanded = tf.TensorArray(labels.dtype, lengths)
    for i in tf.range(1, lengths+1):
        padding_length = tf.expand_dims(lengths, 0) - i
        times_expanded = times_expanded.write(
            i-1,
            tf.concat([times[:i], tf.zeros(tf.concat([padding_length, tf.shape(times)[1:]], 0))], axis=0)
        )
        values_expanded = values_expanded.write(
            i-1,
            tf.concat([values[:i], tf.zeros(tf.concat([padding_length, tf.shape(values)[1:]], 0))], axis=0)
        )
        measurements_expanded = measurements_expanded.write(
            i-1,
            tf.concat([measurements[:i], tf.zeros(tf.concat([padding_length, tf.shape(measurements)[1:]], 0), dtype=bool)], axis=0)
        )
        lengths_expanded = lengths_expanded.write(i-1, i)
        labels_expanded = labels_expanded.write(
            i-1,
            tf.concat([labels[:i], tf.fill(tf.concat([padding_length, tf.shape(labels)[1:]], 0), -100)], axis=0)
        )

    times_expanded = times_expanded.stack()
    values_expanded = values_expanded.stack()
    measurements_expanded = measurements_expanded.stack()
    lengths_expanded = lengths_expanded.stack()
    labels_expanded = labels_expanded.stack()
    return (
        tf.tile(index[None], [lengths]),
        (
            demo_expanded,
            times_expanded,
            values_expanded,
            measurements_expanded,
            lengths_expanded,
        ),
        labels_expanded
    )


def build_reducer(in_dataset):
    shapes = tf.compat.v1.data.get_output_shapes(in_dataset)

    def init_fn(_):
        types = tf.compat.v1.data.get_output_types(in_dataset)
        return (
            tf.constant(0),
            tuple(tf.TensorArray(dtype=d, size=1, dynamic_size=True) for d in types[1]),
            tf.TensorArray(dtype=types[2], size=1, dynamic_size=True)
        )

    def reduce_fn(state, inputs):
        i, data_array, labels = state
        new_i = i+1
        new_data = []
        new_data = tuple(d.write(i, new_input) for d, new_input in zip(data_array, inputs[1]))
        new_labels = labels.write(i, inputs[2])
        return (
            new_i,
            new_data,
            new_labels
        )

    def finalize_fn(final_i, data, labels):
        out = tuple(d.stack() for d in data), labels.stack()

        def fix_shape(tensor_or_array, shape_def):
            if isinstance(tensor_or_array, tuple):
                return tuple(fix_shape(el, shape) for el, shape in zip(tensor_or_array, shape_def))
            else:
                tensor_or_array.set_shape([None] + list(shape_def))
                return tensor_or_array
        return fix_shape(out, shapes[1:])

    return Reducer(init_fn, reduce_fn, finalize_fn)


def evaluate_on_physionet_2019(dataset_name, model, split='validation'):
    if split == 'validation':
        dataset = tfds.load(
            dataset_name, split=tfds.Split.VALIDATION, as_supervised=True)
    elif split == 'testing':
        dataset = tfds.load(
            dataset_name, split=tfds.Split.TEST, as_supervised=True)
    else:
        raise ValueError()

    normalizer = Normalizer(dataset_name)
    task = DATASET_TO_TASK_MAPPING[dataset_name]

    def custom_batch_fn(index, inputs):
        ts, labels = inputs
        return expand_time_into_batch(index, ts, labels)

    def model_preprocessing(index, ts, labels):
        # Remove the padding again
        length = ts[-1]
        demo = ts[0]
        ts = (demo,) + tuple(el[:length] for el in ts[1:-1]) + (length,)
        labels = labels[:length]
        normalized_ts, labels = \
            normalizer.get_normalization_fn()(ts, labels)
        prepro_fn = model.data_preprocessing_fn()
        if prepro_fn is None:
            return index, normalized_ts, labels
        return prepro_fn(normalized_ts, labels) + (index,)

    # Add id to elements in dataset, expand timepoints into fake batch
    # dimension. Remove it and apply model specific preprocessing on the
    # instances.
    preprocessed_expanded = dataset.enumerate().map(
        custom_batch_fn).unbatch().map(model_preprocessing)

    batched_dataset = preprocessed_expanded.padded_batch(
        32,
        get_output_shapes(preprocessed_expanded),
        padding_values=get_padding_values(get_output_types(preprocessed_expanded)),
        drop_remainder=False
    )

    predictions = []
    labels = []
    instance_ids = []
    for instance in tqdm(tfds.as_numpy(batched_dataset)):
        last_index = instance[0][-1] - 1
        instance_predictions = model.predict_on_batch(instance[0])
        instance_labels = instance[1]
        instance_ids.append(instance[2])
        batch_index = np.arange(len(instance_labels))
        predictions.append(instance_predictions[(batch_index, last_index)])
        labels.append(instance_labels[(batch_index, last_index)])
    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)
    instance_ids = np.concatenate(instance_ids, axis=0).ravel()
    instances, indexes = np.unique(instance_ids, return_index=True)
    predictions = np.split(predictions, indexes[1:])
    labels = np.split(labels, indexes[1:])

    return {
        metric_name: metric_fn(labels, predictions)
        for metric_name, metric_fn in task.metrics.items()
    }


def evaluate_model(dataset, model, split):
    if dataset == 'physionet2019':
        return evaluate_on_physionet_2019(dataset, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        required=True,
        type=str,
        choices=DATASET_TO_TASK_MAPPING.keys()
    )
    parser.add_argument(
        '--model',
        required=True,
        type=str,
        choices=seft.models.__all__
    )
    parser.add_argument('--model-config', required=True, type=str)
    parser.add_argument('--model-weights', required=True, type=str)
    parser.add_argument(
        '--splits',
        required=True, nargs='+', type=str, choices=['validation', 'testing'],
    )
    parser.add_argument('--output', required=False, type=str, default=None)

    args = parser.parse_args()

    model = load_model(
        args.dataset, args.model, args.model_config, args.model_weights)
    output = {}
    for split in args.splits:
        out = evaluate_model(args.dataset, model, split)
        out = {
            f'{split}_{metric_name}': metric_value
            for metric_name, metric_value in out.items()
        }
        output.update(out)
    print(output)
    if args.output is not None:
        with open(args.output, 'w') as f:
            json.dump(output, f)
