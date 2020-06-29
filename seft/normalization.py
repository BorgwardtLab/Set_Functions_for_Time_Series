"""Normalization of datasets."""
from collections import defaultdict
from collections.abc import Sequence
import json
import os

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import tensorflow_datasets as tfds

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'resources')


class Normalizer:
    """Normalizer for medical time series datasets."""

    def __init__(self, dataset_name):
        """Initialize normalizer

        Args:
            dataset_name: Name of medical_ts_dataset, loaded via tensorflow
            datasets.

        """
        self.dataset = dataset_name
        self._means = None
        self._stds = None
        self._fields = None

        self._ts_sum_x = None
        self._ts_sum_sq_x = None
        self._ts_count = None
        self._demo_sum_x = None
        self._demo_sum_sq_x = None
        self._demo_count = None
        self._class_balance = defaultdict(lambda: 0)
        self._load_or_compute_normalization()

    def _load_or_compute_normalization(self):
        train_split, info = tfds.load(
            self.dataset,
            split=tfds.Split.TRAIN,
            as_supervised=True,
            with_info=True
        )
        config_file = os.path.join(
            CONFIG_PATH, f'normalization_{info.name}_v{info.version}.json')
        if not os.path.exists(config_file):
            cat_demo = info.metadata['demographics_categorical_indicator']
            cat_ts = info.metadata['combined_categorical_indicator']
            self._compute_parameters_from_split(train_split)
            self._save_params(config_file, cat_demo, cat_ts)
        self.load_params(config_file)

    def _compute_parameters_from_split(self, split):
        np_dataset = tfds.as_numpy(split)
        iterator = tqdm(
            np_dataset,
            unit=' examples',
            desc='Computing normalization statistics'
        )
        for instance in iterator:
            self._feed_instance(instance)

    def _feed_instance(self, inputs):
        demographics, _, ts, measurements = inputs[0][:4]
        # Small hack to get this running
        label = inputs[1]
        try:
            if isinstance(label, Sequence) or isinstance(label, np.ndarray):
                label = np.amax(label)
            self._class_balance[int(label)] += 1
        except Exception:
            # For any other case than classifciation this should fail
            pass

        # x = np.array(x)
        if self._ts_count is None:
            # Initialize variables
            self._ts_count = np.zeros(ts.shape[-1])
            self._ts_sum_x = np.zeros(ts.shape[-1])
            self._ts_sum_sq_x = np.zeros(ts.shape[-1])
            self._demo_count = 0
            self._demo_sum_x = np.zeros(demographics.shape[-1])
            self._demo_sum_sq_x = np.zeros(demographics.shape[-1])

        self._ts_count += np.sum(np.isfinite(ts), axis=0)
        self._ts_sum_x += np.nansum(ts, axis=0)
        self._ts_sum_sq_x += np.nansum(ts**2, axis=0)
        self._demo_count += 1
        self._demo_sum_x += demographics
        self._demo_sum_sq_x += demographics**2

    def _save_params(self, save_file_path, cat_demo, cat_ts):
        n_instances = sum(self._class_balance.values())
        self._class_balance = {
            class_label: n_class/n_instances
            for class_label, n_class in self._class_balance.items()
        }

        eps = 1e-7
        N = self._ts_count
        self._ts_means = self._ts_sum_x / N
        self._ts_stds = np.sqrt(
            1.0/(N - 1) *
            (
                self._ts_sum_sq_x
                - 2.0 * self._ts_sum_x * self._ts_means
                + N * self._ts_means**2
            )
        )
        self._ts_stds[self._ts_stds < eps] = eps
        # Dont normalize categorical variables
        self._ts_means[cat_ts] = 0.
        self._ts_stds[cat_ts] = 1.
        N = self._demo_count
        self._demo_means = self._demo_sum_x / N
        self._demo_stds = np.sqrt(
            1.0/(N - 1) *
            (
                self._demo_sum_sq_x
                - 2.0 * self._demo_sum_x * self._demo_means
                + N * self._demo_means**2
            )
        )
        self._demo_stds[self._demo_stds < eps] = eps
        # Dont normalize categorical variables
        self._demo_means[cat_demo] = 0.
        self._demo_stds[cat_demo] = 1.

        with open(save_file_path, "w") as save_file:
            json.dump(
                {
                    'ts_means': self._ts_means.tolist(),
                    'ts_stds': self._ts_stds.tolist(),
                    'demo_means': self._demo_means.tolist(),
                    'demo_stds': self._demo_stds.tolist(),
                    'class_balance': self._class_balance
                },
                save_file
            )

    def load_params(self, load_file_path):
        """Load normalization parameters from json file.

        Args:
            load_file_path: Path to json file

        """
        with open(load_file_path, "r") as load_file:
            dct = json.load(load_file)
            self._ts_means = np.array(dct['ts_means'])
            self._ts_stds = np.array(dct['ts_stds'])
            self._demo_means = np.array(dct['demo_means'])
            self._demo_stds = np.array(dct['demo_stds'])
            self._class_balance = dct['class_balance']

    def get_normalization_fn(self):
        """Get functions which can be mapped to dataset."""
        def normalization_fn(*inputs):
            features, label = inputs

            def normalize_demo(demo):
                return (demo - self._demo_means) / self._demo_stds

            def normalize_ts(ts, measurements):
                normalized = (ts - self._ts_means) / self._ts_stds
                # Fill nans with zeros
                normalized = tf.where(
                    measurements,
                    normalized,
                    tf.zeros_like(normalized)
                )
                return normalized, measurements

            normalized_features = (
                (normalize_demo(features[0]), ) + features[1:2] +
                normalize_ts(features[2], features[3]) + features[4:]
            )

            return normalized_features, label
        return normalization_fn
