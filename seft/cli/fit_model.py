"""Fit a model in SeFT using a tensorflow dataset."""
import os
import logging

import seft.cli.silence_warnings
# import tensorflow as tf
# tf.enable_eager_execution()
# tf.config.experimental_run_functions_eagerly(True)

import tensorboard.plugins.hparams.api as hp

import seft.models
from seft.training_routine import TrainingLoop
from seft.tasks import DATASET_TO_TASK_MAPPING
from seft.training_utils import LogRealInterval

from .cli_utils import (
    parse_commandline_arguments,
    get_reproducable_commandline,
    save_args_to_json
)
from .hyperparameters import get_hyperparameter_settings

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handle_special_cases(args, hyperparameters):
    """Evtl. restrict hyperparameter search space based on model or dataset."""
    def hyperparameter_by_name(name):
        for h in hyperparameters:
            if h.name == name:
                return h
        raise IndexError()

    if args.model in ['InterpolationPredictionModel', 'MGPGRUModel']:
        # The InterpolationPredictionModel has high requirements for memory,
        # thus we need to upper bound the batch size when running this model
        batch_size = hyperparameter_by_name('batch_size')
        batch_size._domain = hp.Discrete([16, 32, 64])
        batch_size._default = 32
        if args.dataset == 'physionet2019':
            # The physionet2019 dataset has higher memory requirements, thus bs
            # of 64 is not possible for these models
            batch_size._domain = hp.Discrete([16, 32])

    if args.model == 'TransformerModel':
        warmup = hyperparameter_by_name('warmup_steps')
        warmup._domain = hp.Discrete([1000])
        warmup._default = 1000
        if args.dataset in ['physionet2019']:
            # We dont support max aggregation for online scenarios for now
            aggregation_fn = hyperparameter_by_name('aggregation_fn')
            aggregation_fn._domain = hp.Discrete(['mean', 'sum'])

    if args.model == 'RevisedDeepSetAttentionModel3':
        batch_size = hyperparameter_by_name('batch_size')
        batch_size._domain = hp.Discrete([64, 128])
        learning_rate = hyperparameter_by_name('learning_rate')
        learning_rate._domain = LogRealInterval(0.0005, 0.001)

    return hyperparameters


def create_subfolder(parent_folder, max_number=9999):
    """Create folder with consequtive number in the parent_folder.

    Avoids race coditions in the creation of the subfolders.

    Args:
        parent_folder: Where to create a new folder

    Returns:
        The path tho the new directory.

    """
    for subfolder_index in range(max_number):
        try:
            new_folder_path = os.path.join(parent_folder, str(subfolder_index))
            os.makedirs(new_folder_path)
            break
        except FileExistsError:
            continue

    return new_folder_path


def set_seed_random_number_generators(seed):
    """Set seeds of random number generators.

    This initializes the RNG of python, numpy and tensorflow to the same seed.

    Args:
        seed: An integer

    """
    import random
    import numpy as np
    import tensorflow as tf
    random.seed(seed.to_bytes(4, byteorder='little', signed=False), version=2)
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.get_default_session())


def main():
    args, hyperparameters = parse_commandline_arguments()

    # General options
    log_dir = None
    if args.log_dir is not None:
        if args.log_dir.endswith('/'):
            log_dir = create_subfolder(args.log_dir)
        else:
            log_dir = args.log_dir
            os.makedirs(log_dir, exist_ok=False)
        logger.debug(f'Saving results to {log_dir}')

    hyperparameters = handle_special_cases(
        args, hyperparameters)
    hyperparameter_dict, args = get_hyperparameter_settings(
        hyperparameters, args)

    # Save the state
    if log_dir is not None:
        args.hypersearch = False
        save_args_to_json(args, os.path.join(log_dir, 'config.json'))

    print('Recreate run using following command:')
    commandline = get_reproducable_commandline(args, hyperparameter_dict)
    print(commandline)
    set_seed_random_number_generators(args.random_seed)
    task = DATASET_TO_TASK_MAPPING[args.dataset]
    model = getattr(seft.models, args.model).from_hyperparameter_dict(
        task, hyperparameter_dict)

    train_loop = TrainingLoop(
        model,
        args.dataset,
        task,
        args.max_epochs,
        hyperparameter_dict,
        args.early_stopping,
        log_dir,
        balance_dataset=args.balance,
        debug=args.debug
    )
    train_loop()


if __name__ == '__main__':
    main()
