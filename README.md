# SeFT - Set functions for Time Series

This is the main source code for the submission `Set Functions for Time
Series`. It depends on two further packages
[keras-transformer](https://github.com/ExpectationMax/keras-transformer)
(fork with support for sequences of different lengths) and
[medical-ts-datasets](https://github.com/ExpectationMax/medical_ts_datasets)
(containing the implementation of the datasets used).

## Installation

The project requires python version `3.7` (newer versions of python are unfortunately
incompatible with tensorflow 1.15.x) and is set up to use `poetry` packaging utility.
The easiest way to get started is to create a virtual python environment and to install
the package inside this environment.

```bash
git clone https://github.com/BorgwardtLab/Set_Functions_for_Time_Series.git
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

### Adding new datasets
The datasets used in this code are implemented in a separate package [medical_ts_dataset](https://github.com/ExpectationMax/medical_ts_datasets).
In total you would need to run the following steps:

 1. Implement dataset in a fork of `medical_ts_datasets` or create you own package with the implementation. The easiest way is probably to adapt one of the readers to fit your format (see the `medical_ts_datasets` package). For some further information I recommend consulting the [tfds documentation](https://www.tensorflow.org/datasets/add_dataset). In the end the following code should be able to run:
   ```python
   import tensorflow_datasets as tfds
   import medical_ts_datasets      # this registers your dataset or any other dataset with tensorflow datasets
   import my_package_with_dataset  # alternatively if you decide to implement you datasets in a separate package
   tfds.load('<your_dataset_name>')
   ```
 2. Add a an entry to the directory [here](https://github.com/BorgwardtLab/Set_Functions_for_Time_Series/blob/d72d446f26c68a3f0f73edb2251e2e55defa5129/seft/tasks.py#L218) defining which type of endpoint you dataset provides.
 3. Optional, if you decide to implement the dataset in a separate package) add import statements to you package [here](https://github.com/BorgwardtLab/Set_Functions_for_Time_Series/blob/d72d446f26c68a3f0f73edb2251e2e55defa5129/seft/training_utils.py#L9)

Then you should be able to run all models in this codebase on your data.
On a final note as I assume you are working with medical data (which is usually not publically accessable on the internet) [this section](https://www.tensorflow.org/datasets/add_dataset#manual_download_and_extraction) of the tfds documentation might come in handy.

## Available models

The following models are supported:
`GRUSimpleModel`, `PhasedLSTMModel`, `InterpolationPredictionModel`,
`GRUDModel`, `TransformerModel`, `DeepSetAttentionModel`

It was brought to my attention, that the GRUD implementation is not entirely
in line with the original paper. For further information please refer to this
[issue](https://github.com/BorgwardtLab/Set_Functions_for_Time_Series/issues/1).

# Copyright

This collective work is copyright (c) 2020 by Max Horn. Individual
portions may be copyright by individual contributors, and are included
in this collective work with permission of the copyright owners.
