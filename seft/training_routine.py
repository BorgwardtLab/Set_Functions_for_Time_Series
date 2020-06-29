"""Training routine for models."""
from os.path import join
import json
from itertools import chain

import numpy as np
import tensorflow as tf
from typing import Callable
from tensorflow.keras.callbacks import (
    CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard)

from .callbacks import (
    EvaluationCallback, HParamsCallback, TimeHistory, WarmUpScheduler)
from .normalization import Normalizer
from .training_utils import (
    build_training_iterator,
    build_validation_iterator,
    build_test_iterator,
    build_hyperparameter_metrics,
    init_hyperparam_space,
    make_one_shot_iterator
)
from .logging_utils import NumpyEncoder


class TrainingLoop(Callable):
    def __init__(self, model, dataset, task, epochs, hparams, early_stopping,
                 rundir, balance_dataset=True, additional_callbacks=None,
                 debug=False):
        self.model = model
        self.dataset = dataset
        self.task = task
        self.normalizer = Normalizer(dataset)
        self.n_epochs = epochs
        self.early_stopping = early_stopping
        self.hparams = hparams
        self.rundir = rundir
        self.balance_dataset = balance_dataset
        self.debug = debug

        self.callbacks = self._build_callbacks(additional_callbacks)

    def _build_callbacks(self, additional_callbacks):
        callbacks = []
        # Time per epoch callback
        time_cb = TimeHistory()
        callbacks.append(time_cb)

        # Evaluation callback
        # Repeat epochs + 1 times as we run an additional validation step at
        # the end of training afer recovering the model.
        val_iterator_cb, _ = build_validation_iterator(
            self.dataset,
            self.hparams['batch_size'],
            self._normalize_and_preprocess()
        )
        val_cb = EvaluationCallback(
            val_iterator_cb,
            'val',
            metrics=self.task.metrics
        )
        callbacks.append(val_cb)

        # Early stopping callback
        early_stopping_cb = EarlyStopping(
            'val_' + self.task.monitor_quantity,
            mode=self.task.direction_of_improvement,
            patience=self.early_stopping,
            min_delta=0.0001
        )
        callbacks.append(early_stopping_cb)

        # LR scheduling
        reduce_on_plateau_cb = ReduceLROnPlateau(
            monitor='val_' + self.task.monitor_quantity,
            factor=0.5,
            patience=self.early_stopping // 2,
            verbose=1,
            mode=self.task.direction_of_improvement,
            min_delta=0.0001,
            cooldown=self.early_stopping // 2,
            min_lr=1e-5
        )
        callbacks.append(reduce_on_plateau_cb)
        callbacks.append(WarmUpScheduler(
            self.hparams['learning_rate'],
            warmup_steps=self.hparams['warmup_steps'],
            verbose=1
        ))

        # Logging callbacks
        metrics = build_hyperparameter_metrics(self.task.metrics)
        if self.rundir:
            with open(join(self.rundir, 'model.json'), 'w') as f:
                json.dump(self.model.get_config(), f)

            init_hyperparam_space(
                join(self.rundir, 'tb'),
                self.hparams.get_hyperparameter_mapping().keys(),
                metrics
            )
            callbacks.append(CSVLogger(join(self.rundir, 'metrics.csv')))
            callbacks.append(ModelCheckpoint(
                join(self.rundir, 'model_weights.hdf5'),
                save_best_only=True,
                save_weights_only=True,
                monitor='val_' + self.task.monitor_quantity,
                mode=self.task.direction_of_improvement
            ))
            callbacks.append(
                TensorBoard(
                    join(self.rundir, 'tb'),
                    update_freq=5,
                    histogram_freq=5
                ))
            callbacks.append(
                HParamsCallback(
                    join(self.rundir, 'tb'),
                    dict(chain(
                        {
                            'dataset': self.dataset,
                            'model': self.model.__class__.__name__
                        }.items(),
                        self.hparams.get_hyperparameter_mapping().items()
                    ))
                )
            )

        if additional_callbacks is not None:
            callbacks.expand(additional_callbacks)
        return callbacks

    def _normalize_and_preprocess(self, with_weights=False):
        """Normalize input data and apply model specific preprocessing fn."""
        if self.model.data_preprocessing_fn() is None:
            return self.normalizer.get_normalization_fn()

        def combined_fn(ts, labels):
            normalized_ts, labels = \
                self.normalizer.get_normalization_fn()(ts, labels)
            preprocessed_ts, labels = \
                self.model.data_preprocessing_fn()(normalized_ts, labels)

            # No class weights
            if self.task.class_weights is None or with_weights is False:
                return preprocessed_ts, labels

            # Evtl. use class weights (until now only relevant for
            # physionet2019).
            class_weights = self.task.class_weights
            weights = \
                tf.constant([class_weights[i] for i in range(len(class_weights))])
            sample_weights = tf.gather(weights, tf.reshape(labels, (-1, )), axis=0)
            sample_weights = tf.reshape(sample_weights, tf.shape(labels)[:-1])
            return preprocessed_ts, labels, sample_weights

        return combined_fn

    def _prepare_dataset_for_training(self):
        if self.balance_dataset:
            class_balance = [
                self.normalizer._class_balance[str(i)] for i in range(2)]
        else:
            class_balance = None
        train_iterator, train_steps = build_training_iterator(
            self.dataset,
            self.n_epochs,
            self.hparams['batch_size'],
            self._normalize_and_preprocess(with_weights=True),
            balance=self.balance_dataset,
            class_balance=class_balance
        )
        # Repeat epochs + 1 times as we run an additional validation step at
        # the end of training afer recovering the model.
        val_iterator, val_steps = build_validation_iterator(
            self.dataset,
            self.hparams['batch_size'],
            self._normalize_and_preprocess()
        )
        return train_iterator, train_steps, val_iterator, val_steps

    def _get_train_metrics(self, history):
        train_metrics = {
            f'train_final_{metric}': history.history[metric][-1]
            for metric in history.history.keys()
            if not metric.startswith('val') and metric != 'lr'
        }
        if self.task.direction_of_improvement == 'min':
            best_index = np.argmin(history.history['val_' + self.task.monitor_quantity])
        elif self.task.direction_of_improvement == 'max':
            best_index = np.argmax(history.history['val_' + self.task.monitor_quantity])
        else:
            raise ValueError()

        train_metrics.update({
            f'train_restored_{metric}': history.history[metric][best_index]
            for metric in history.history.keys()
            if not metric.startswith('val') and metric != 'lr'
        })
        return train_metrics

    def _restore_and_evaluate_model(self, val_iter):
        if self.rundir:
            print(f'Loading model with {self.task.direction_of_improvement} '
                  f'validation {self.task.monitor_quantity}.')
            self.model.load_weights(join(self.rundir, 'model_weights.hdf5'))
        else:
            print(
                'Unable to load best model. No rundir provided.'
            )

        # Evaluate model on validation
        validation_metrics = {}
        self.callbacks[1].on_epoch_end(-1, validation_metrics)
        val_eval = self.model.evaluate(val_iter)
        val_eval = {
            f'val_{name}': value
            for name, value in zip(self.model.metrics_names, val_eval)
        }
        validation_metrics.update(val_eval)

        # Evaluate model on testing
        # Load data and build fake callback
        test_iterator, _ = build_test_iterator(
            self.dataset,
            self.hparams['batch_size'],
            self._normalize_and_preprocess()
        )
        test_cb = EvaluationCallback(
            test_iterator,
            'test',
            metrics=self.task.metrics,
            print_evaluations=False
        )
        test_cb.model = self.model

        test_metrics = {}
        test_cb.on_epoch_end(-1, test_metrics)
        test_eval = self.model.evaluate(test_iterator)
        test_eval = {
            f'test_{name}': value
            for name, value in zip(self.model.metrics_names, test_eval)
        }
        test_metrics.update(test_eval)
        return validation_metrics, test_metrics

    def _add_metrics_to_tensorboard(self, train_metrics, val_metrics,
                                    test_metrics):
        if self.rundir:
            tf_dir = join(self.rundir, 'tb')
            sess = tf.compat.v1.keras.backend.get_session()
            with tf.compat.v2.summary.create_file_writer(tf_dir).as_default() as w:
                sess.run(w.init())
                for key, value in train_metrics.items():
                    sess.run(
                        tf.compat.v2.summary.scalar(key, data=value, step=0))
                sess.run(w.flush())

                for key, value in val_metrics.items():
                    sess.run(
                        tf.compat.v2.summary.scalar(
                            'best_' + key, data=value, step=0))
                sess.run(w.flush())

                for key, value in test_metrics.items():
                    sess.run(
                        tf.compat.v2.summary.scalar(key, data=value, step=0))
                sess.run(w.flush())

    def _build_results_summary(self, history, train_metrics, val_metrics,
                               test_metrics):
        result = {}
        result['history'] = history.history
        result['mean_epoch_time'] = self.callbacks[0].get_average_epoch_time()

        result.update(train_metrics)
        result.update(val_metrics)
        result.update(test_metrics)

        result['hyperparameters'] = {
            h.name: value
            for h, value in self.hparams.get_hyperparameter_mapping().items()
        }
        result['max_epochs'] = self.n_epochs
        result['early_stopping'] = self.early_stopping
        return result

    def _save_result(self, result):
        print(result)
        if self.rundir:
            with open(join(self.rundir, 'results.json'), 'w') as f:
                json.dump(result, f, cls=NumpyEncoder, indent=2)

    def __call__(self):
        if self.debug:
            from tensorflow.python import debug as tf_debug
            sess = tf.keras.backend.get_session()
            sess = tf_debug.LocalCLIDebugWrapperSession(
                sess, ui_type="readline")
            tf.keras.backend.set_session(sess)

        train_iter, steps_per_epoch, val_iter, val_steps = \
            self._prepare_dataset_for_training()

        self.model.compile(
            optimizer='adam',
            lr=self.hparams['learning_rate'],
            loss=self.task.loss,
            metrics=['accuracy'],
            # TODO: Continue here
            sample_weight_mode=(
                None if self.task.class_weights is None else "temporal")
        )
        history = self.model.fit(
            train_iter,
            epochs=self.n_epochs,
            callbacks=self.callbacks,
            steps_per_epoch=steps_per_epoch,
            # Pass a iterator over dataset with repeat, otherwise the cache is
            # reset after each epoch. This has the disadvantage that we need to
            # also pass validation_steps.
            validation_data=val_iter,
            validation_steps=val_steps,
            verbose=1
        )

        # Eval and summary
        train_metrics = self._get_train_metrics(history)
        val_metrics, test_metrics = \
            self._restore_and_evaluate_model(val_iter)
        self._add_metrics_to_tensorboard(
            train_metrics, val_metrics, test_metrics)
        result = self._build_results_summary(
            history, train_metrics, val_metrics, test_metrics)
        self._save_result(result)
        print('Successfully completed.')
