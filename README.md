# SeFT - Set functions for Time Series

This is the main source code for the submission `Set Functions for Time
Series`. It depends on two further packages
[keras-transformer](https://github.com/ExpectationMax/keras-transformer)
(fork with support for sequences of different lengths) and
[medical-ts-datasets](https://github.com/ExpectationMax/medical_ts_datasets)
(containing the implementation of the datasets used) which are included into
the repository in the `repos` directory as git submodules.

## Installation

The project requires at least python version `3.7` and is set up to use
`poetry` packaging utility. The easiest way to get started is to create
a virtual python environment and to install the package inside this
environment.

```bash
# Clone the repository using the `--recursive` option
git clone --recursive https://github.com/BorgwardtLab/Set_Functions_for_Time_Series.git
cd Set_Functions_for_Time_Series
# Install poetry
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
# Install the `SeFT` package with its dependencies
poetry install
```

## Quickstart

One can quickly test the models on one of the online available datasets. If
they are not yet downloaded, they will be automatically downloaded and stored
in the directory `~/tensorflow_datasets`.

Example usage (in SeFT subdirectory):
```bash
$ poetry run seft_fit_model --dataset physionet2019 --balance --log_dir test_transformer TransformerModel 

Recreate run using following command:
seft_fit_model --random_seed 982927477 --dataset physionet2019 --balance \
  --max_epochs 300 --early_stopping 30 --log_dir test_transformer \
  TransformerModel --learning_rate 0.001 --batch_size 64 --warmup_steps 1000 \
  --n_dims 128 --n_heads 4 --n_layers 1 --dropout 0.0 --attn_dropout 0.0 \
  --aggregation_fn mean --max_timescale 100.0
Train on 176 steps, validate on 101 steps
Epoch 1/300
  5/176 [..............................] - ETA: 8:18 - loss: 0.1676 - acc: 0.1056 
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

# Copyright

This collective work is copyright (c) 2020 by Max Horn. Individual
portions may be copyright by individual contributors, and are included
in this collective work with permission of the copyright owners.
