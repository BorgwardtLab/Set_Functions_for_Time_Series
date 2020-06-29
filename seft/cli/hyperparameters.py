"""Handling of hyperparameters for training."""

import tensorboard.plugins.hparams.api as hp

from seft.training_utils import HParamWithDefault, LogRealInterval


def training_hyperparameters():
    """Get list of training hyperparameters valid for all models."""
    learning_rate = HParamWithDefault(
        'learning_rate',
        LogRealInterval(0.0001, 0.01),
        default=0.001
    )
    # Only vary up to 128 as we otherwise get GPU memory issues on HealingMNIST
    # and PhysionetInHospitalMortality
    batch_size = HParamWithDefault(
        'batch_size',
        hp.Discrete([32, 64, 128, 256, 512]),
        default=64
    )

    warmup = HParamWithDefault(
        'warmup_steps',
        hp.Discrete([0]),
        default=0
    )
    return [learning_rate, batch_size, warmup]


def get_hyperparameter_settings(hyperparameters, args):
    """Read or sample hyperparameters based on argparse Namespace.

    If args.hypersearch is True sample hyperparameters from domain, otherwise
    use default value. Hyperparameter settings defined via the command line
    will never be sampled.

    Also returns an argparse Namespace with the hyerparameters set to the evtl.
    sampled values in order to recreate run.

    Args:
        hyperparameters: List of hyperparameter objects
        args: Argparse Namespace

    Returns:
        HyperparameterDict containing name -> value mapping
        argparse Namespace

    """
    hyperparameter_dict = HyperparameterDict()
    for param in hyperparameters:
        setting = getattr(args, param.name, UndefinedHyperparameter())
        if isinstance(setting, UndefinedHyperparameter):
            if args.hypersearch:
                value = param.domain.sample_uniform()
            else:
                value = param.default
            hyperparameter_dict[param] = value
            setattr(args, param.name, value)
        else:
            hyperparameter_dict[param] = setting
    return hyperparameter_dict, args


class HyperparameterDict(dict):
    """Subclass of dict but allows to acces hyperparameters by name.

    Should contain mappings from hp.HParam to a value, but overrides
    __getitem__ such that it is possible to acces the value of
    a hyperparameter using its name.
    """

    def __getitem__(self, name):
        """Get hyperparameter value by name.

        Args:
            name: Name of hyperparameter

        Returns:
            Value of hyperparameter

        """
        for h in self.keys():
            if h.name == name:
                return super().__getitem__(h)
        raise IndexError()

    def get_hyperparameter_mapping(self):
        """Get dictionary containing the map from hp.HParam to values."""
        return super()

    def __str__(self):
        return str({h.name: value for h, value in super().items()})


class UndefinedHyperparameter:
    """Represents a hyperparameter which was not defined by the cli."""
