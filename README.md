# SeFT - Set functions for Time Series

This is the main source code for the submission `Set Functions for Time
Series`. It depends on two further packages `keras-transformer` (fork with
support for sequences of different lengths) and `medical-ts-datasets`
( containing the implementation of the datasets used) which are in the parent
directory.


## Installation

The project requires at least python version `3.7` and is set up to use
`poetry` packaging utility. The easiest way to get started is to create
a virtual python environment and to install the package inside this
environment.

```bash
cd SeFT_source_submission  # or `cd ..` if you are in the folder of this readme
# Install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
# Install the `SeFT` package with its dependencies
cd SeFT && poetry install
```

## Quickstart

One can quickly test the models on one of the online available datasets. If
they are not yet downloaded, they will be automatically downloaded and stored
in the directory `~/tenosrflow_datasets`.

Example usage (in SeFT subdirectory):
```bash
poetry run seft_fit_model --dataset physionet2019 --balance --log_dir test_transformer TransformerModel 
```

## Usage
```bash
$ poetry run seft_fit_model --help
usage: seft_fit_model [-h] [--max_epochs MAX_EPOCHS]
                      [--random_seed RANDOM_SEED] [--debug]
                      [--log_dir LOG_DIR] [--early_stopping EARLY_STOPPING]
                      [--dataset {physionet2012,physionet2019,mimic3_mortality,mimic3_phenotyping}]
                      [--balance] [--hypersearch]
                      {config,GRUSimpleModel,PhasedLSTMModel,InterpolationPredictionModel,GRUDModel,TransformerModel,DeepSetAttentionModel}
                      ...

positional arguments:
  {config,GRUSimpleModel,PhasedLSTMModel,InterpolationPredictionModel,GRUDModel,TransformerModel,DeepSetAttentionModel}

optional arguments:
  -h, --help            show this help message and exit
  --max_epochs MAX_EPOCHS
  --random_seed RANDOM_SEED
  --debug
  --log_dir LOG_DIR     Where to log results. If ends on backslash assume we
                        need to create a directory
  --early_stopping EARLY_STOPPING
  --dataset {physionet2012,physionet2019,mimic3_mortality,mimic3_phenotyping}
  --balance        Balance the dataset
  --hypersearch    Sample hyperparameters

```

For each model all hyperparameters can be set using the command line:
```bash
poetry run seft_fit_model GRUDModel --help
usage: seft_fit_model GRUDModel [-h] [--learning_rate LEARNING_RATE]
                                [--batch_size BATCH_SIZE]
                                [--warmup_steps WARMUP_STEPS]
                                [--n_units N_UNITS] [--dropout DROPOUT]
                                [--recurrent_dropout RECURRENT_DROPOUT]

optional arguments:
  -h, --help            show this help message and exit
  --learning_rate LEARNING_RATE
  --batch_size BATCH_SIZE
  --warmup_steps WARMUP_STEPS
  --n_units N_UNITS
  --dropout DROPOUT
  --recurrent_dropout RECURRENT_DROPOUT
```

## Available datasets

While the `physionet2012` and `physionet2019` datasets are publicly available
and thus can be automatically downloaded, this is not the case for the
`mimic3_mortality` dataset. Here, after fulfilling the requirements for access
the data needs to be manually downloaded and provided in the form of
a compressed file. We are planning to make this file available in the MIMIC-III
preprocessed data repository.


## Available models

The following models are supported:
`GRUSimpleModel`, `PhasedLSTMModel`, `InterpolationPredictionModel`,
`GRUDModel`, `TransformerModel`, `DeepSetAttentionModel`
