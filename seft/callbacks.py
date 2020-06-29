"""Module containing Keras callbacks."""
import time
from itertools import chain

import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorboard.plugins.hparams.api as hp
from tensorflow.keras.callbacks import TensorBoard


class FailsafeTensorBoard(TensorBoard):
    """Failsafe version of Tensorboard logger."""
    def on_epoch_begin(self, epoch, logs=None):
        try:
            super().on_epoch_begin(epoch, logs)
        except Exception as e:
            print('Got exception in TensorBoard callback:', e)

    def on_epoch_end(self, epoch, logs=None):
        try:
            super().on_epoch_begin(epoch, logs)
        except Exception as e:
            print('Got exception in TensorBoard callback:', e)

    def on_batch_start(self, batch, logs=None):
        try:
            super().on_epoch_begin(batch, logs)
        except Exception as e:
            print('Got exception in TensorBoard callback:', e)

    def on_batch_end(self, batch, logs=None):
        try:
            super().on_epoch_begin(batch, logs)
        except Exception as e:
            print('Got exception in TensorBoard callback:', e)



class WarmUpScheduler(tf.keras.callbacks.Callback):
    def __init__(self, final_lr, warmup_learning_rate=0.0, warmup_steps=0,
                 verbose=0):
        """Constructor for warmup learning rate scheduler.

        Args:
            learning_rate_base: base learning rate.
            warmup_learning_rate: Initial learning rate for warm up. (default:
                0.0)
            warmup_steps: Number of warmup steps. (default: 0)
            verbose: 0 -> quiet, 1 -> update messages. (default: {0})

        """

        super().__init__()
        self.final_lr = final_lr
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.verbose = verbose

        # Count global steps from 1, allows us to set warmup_steps to zero to
        # skip warmup.
        self.global_step = 1
        self._increase_per_step = \
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.global_step <= self.warmup_steps:
            increase = \
                (self.final_lr - self.warmup_learning_rate) / self.warmup_steps
            new_lr = self.warmup_learning_rate + (increase * self.global_step)
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                print(
                    f'Warmup - learning rate: '
                    f'{new_lr:.6f}/{self.final_lr:.6f}',
                    end=''
                )


class EvaluationCallback(tf.keras.callbacks.Callback):
    """Evaluate model on the provided dataset."""

    def __init__(self, dataset, name, metrics={}, print_evaluations=True):
        """Initialize evaluation callback.

        Args:
            dataset: The dataset, should be a tensorflow dataset outputting
                tuples of (X, y).
            name: Name to prepend metric name in log.
            metrics: Dictionary of metrics {name: function(y_true, pred)}.
            print_evaluations: Print the result of the evaluations.

        """
        super().__init__()
        self.dataset = dataset
        self.dataset_name = name
        self.metrics = metrics
        self.print_evaluations = print_evaluations
        label_iter = dataset.map(lambda data, labels: labels)

        label_batches = list(tfds.as_numpy(label_iter))

        if label_batches[0].ndim == 3:
            # Online prediction scenario
            def remove_padding(label_batch):
                # Online prediction scenario
                labels = []
                for instance in label_batch:
                    is_padding = np.all((instance == -100), axis=-1)
                    labels.append(instance[~is_padding])
                return labels

            self.labels = list(chain.from_iterable(
                [
                    remove_padding(label_batch)
                    for label_batch in label_batches
                ]
            ))
            self.online = True
        else:
            # Whole time series classification scenario
            self.labels = np.concatenate(label_batches, axis=0)
            self.online = False

        self.data_iterator = dataset

    def on_epoch_end(self, epoch, log={}):
        """Run evaluations."""

        if self.online:
            batch_predictions = []
            def get_data(d, l):
                return d
            for batch in tfds.as_numpy(self.data_iterator.map(get_data)):
                batch_predictions.append(self.model.predict_on_batch(batch))

            predictions = chain.from_iterable(batch_predictions)
            # Split off invalid predictions
            predictions = [
                prediction[:len(label)]
                for prediction, label in zip(predictions, self.labels)
            ]
        else:
            predictions = self.model.predict(self.data_iterator)

        if self.print_evaluations:
            print()
        for metric_name, metric_fn in self.metrics.items():
            score = metric_fn(self.labels, predictions)
            if self.print_evaluations:
                print(f'Epoch {epoch+1}: {self.dataset_name} {metric_name}: '
                      f'{score:5.5f}')
            log[f'{self.dataset_name}_{metric_name}'] = score
        return False


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, ignore_first=1, logs={}):
        self.ignore_first = ignore_first
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

    def get_average_epoch_time(self):
        # Ignore first epoch as the dataset still needs to be cached
        return float(np.mean(self.times[self.ignore_first:]))


class HParamsCallback(tf.keras.callbacks.Callback):
    def __init__(self, logdir, hparams):
        self._writer = tf.compat.v2.summary.create_file_writer(logdir)
        self._hparams = hparams

    def on_train_begin(self, logs=None):
        sess = tf.compat.v1.keras.backend.get_session()
        with self._writer.as_default() as w:
            sess.run(w.init())
            sess.run(hp.hparams(self._hparams))
            sess.run(w.flush())


