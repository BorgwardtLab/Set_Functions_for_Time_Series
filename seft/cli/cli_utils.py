"""Handling of commandline arguments."""
import argparse
import json
import os
import sys

import medical_ts_datasets
import seft.models

from .hyperparameters import training_hyperparameters, UndefinedHyperparameter


def generate_random_seed():
    """Generate a random 32 bit unsigned integer as seed."""
    rand_bytes = os.urandom(4)
    return int.from_bytes(rand_bytes, byteorder='little', signed=False)


def add_hyperparameters_to_parser(parser, hparams):
    """Create arguments in Argument parser with correspond to hyperparameters.

    Args:
        parser: Argparse Argument parser
        hparams: List of Hyperparameters

    Returns:
        Parser with added arguments

    """
    for hparam in hparams:
        parser.add_argument(
            '--' + hparam.name,
            type=type(hparam.default),
            default=UndefinedHyperparameter()
        )
    return parser


def get_reproducable_commandline(args, hyperparameters):
    """Print the commandline that can be used to reproduce this run.

    Args:
        args: Argparse namespace
        hyperparameters: Hyperparameter dict of Hyperparameter -> value
            mapping.

    """
    logdir = f'--log_dir {args.log_dir} ' if args.log_dir is not None else ''
    balance = '--balance ' if args.balance else ''

    command = (
        f'{sys.argv[0]} --random_seed {args.random_seed} '
        f'--dataset {args.dataset} {balance}'
        f'--max_epochs {args.max_epochs} '
        f'--early_stopping {args.early_stopping} '
        f'{logdir} {args.model} '
    )
    for hparam, value in hyperparameters.items():
        command += f'--{hparam.name} {value} '
    return command


def save_args_to_json(args, path):
    """Save the argparse namespace to a json file.

    Args:
        args: The argparse namespace to save.
        path: Path to json file for storage.

    """
    args_dict = dict(**vars(args))
    with open(path, 'w') as f:
        json.dump(args_dict, f)


def load_args_from_json(args, path):
    """Load argparse namespace from json file.

    Args:
        args: The argparse namespace to save.
        path: Path to json file for storage.

    """
    with open(path, 'r') as f:
        arguments_dict = json.load(f)
    for name, value in arguments_dict.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    return args


def parse_commandline_arguments():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=300)
    parser.add_argument(
        '--random_seed', type=int, default=generate_random_seed())
    parser.add_argument(
        '--debug', default=False, action='store_true')
    parser.add_argument(
        '--log_dir', type=str, default=None,
        help='Where to log results. If ends on backslash '
             'assume we need to create a directory'
    )
    parser.add_argument('--early_stopping', default=30, type=int)
    parser.add_argument(
        '--dataset',
        required=False,
        choices=medical_ts_datasets.builders
    )
    parser.add_argument('--balance', default=False, action='store_true')
    parser.add_argument('--hypersearch', default=False, action='store_true')
    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True

    # Alow using config file
    config_parser = subparsers.add_parser('config')
    config_parser.add_argument('config_file', type=str)

    model_hyperparameters = {}
    for model_name in seft.models.__all__:
        model = getattr(seft.models, model_name)
        model_parser = subparsers.add_parser(model_name)
        hyperparameters = training_hyperparameters()
        hyperparameters.extend(model.get_hyperparameters())
        model_hyperparameters[model_name] = hyperparameters
        add_hyperparameters_to_parser(model_parser, hyperparameters)

    args = parser.parse_args()

    # Load config if provided
    if args.model == 'config':
        config_file = args.config_file
        del(args.model)
        del(args.config_file)
        args = load_args_from_json(args, config_file)
        print(f'Loaded config {config_file}: {args}')

    return args, model_hyperparameters[args.model]
