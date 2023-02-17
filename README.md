# Extracting Structured Web Content using Deep Neural Language Models

This repository contains the source code used for the experiments in my Master thesis: [Extracting Structured Web Content using Deep Neural Language Models](https://www.ru.nl/publish/pages/769526/gijs_hendriksen.pdf). The same scripts were also used to generate the numbers in the paper "An Analysis of Information Extraction Models and Their Evaluation".

## Overview

This repository consists of the following parts:

- `information_extraction`: a Python module used for implementing information extraction tools and evaluation metrics. Most notably, it contains the code for training and evaluating a BERT baseline for (closed) information extraction.
- `scripts`: a collection of scripts that use the `information_extraction` package to 1) preprocess SWDE data, 2) train information extraction models, and 3) evaluate the performance of information extraction models
- `sweeps`: the configuration files for Weights & Biases sweeps, that were used to run the experiments with our information extraction models.

For more details on the experiments performed as part of this research, please refer to the Master thesis linked above.

## Installation

To get started, use [Poetry](https://python-poetry.org/) to install all dependencies.

```
$ poetry install
```

Then, either activate the virtual environment to execute all Python code in the virtual environment, or prepend every command with `poetry run`.

```
$ poetry shell
(venv) $ python scripts/preprocessing/perform_data_split.py
```

or:

```
$ poetry run python scripts/preprocessing/perform_data_split.py
```

All commands in this README should be executed from within the virtual environment

## Usage

### Configuration

This project makes heavy use of [Weights & Biases](https://wandb.ai/) (W&B), for logging of data, experiments and results. All intermediate data products are also stored as [Weights & Biases Artifacts](https://docs.wandb.ai/guides/artifacts). To use the code, you first need to connect to W&B:

```
wandb login
```

By default, all data and runs will be stored in a W&B workspace called `information_extraction`. You can change this default in the config file `information_extraction/config.py` (`WANDB_PROJECT`).

To use datasets stored in W&B, they first need to be downloaded to your local machine. The directory in which all data will be stored can also be configured in the config file (`DATA_DIR`). The default directory is `~/Data/`.

### Data setup

To get started, ensure the SWDE and Expanded SWDE datasets are in your `DATA_DIR` (`$DATA_DIR/SWDE` and `$DATA_DIR/ExpandedSWDE`, respectively). Then, upload the SWDE dataset to W&B for use in other parts of the project.

```
wandb artifact put -n swde -d 'The raw SWDE dataset' -t dataset ~/Data/SWDE/
``` 

### Preprocessing

First, we split the SWDE dataset into train/validation/test splits. This is done using the `perform_data_split` script. To get instructions on its usage, issue the command:

```
python scripts/preprocessing/perform_data_split.py -h
``` 

There are three supported data splits:

- `random`: a fully random train/validation/test split of all webpages.
- `zero_shot`: a train/validation/test split that ensures there is no overlap of websites between the splits.
- `webke`: a train/validation/test split reverse-engineered from the preprocessed WebKE data.

With these data splits, we can further preprocess the data into the representations that are used by our extraction models. This is done using:

```
python scripts/preprocessing/perform_preprocessing.py
```

### Training and running experiments

Training is done using the scripts in the `scripts/training` directory. They are executed using [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps) to efficiently perform experiments with varying inputs.

To run the final experiments of our research and reproduce our numbers, you can run the sweeps in `sweeps/final`. For instance:

```
wandb sweep sweeps/final/000-context-size.yaml
```

This initializes the sweep, and gives you a `wandb agent` command to start agents that execute these experiments. To run multiple experiments in parallel on a single machine with multiple GPUs, you can use `screen` to start individual sessions with different values for the `CUDA_VISIBLE_DEVICES` to assign a specific GPU to each session. 

### Analysis

Finally, we can analyse the results of the experiments we just performed, using the scripts in `scripts/analysis`. Each of the experiments/sweeps in `sweeps/final` has a dedicated analysis script to generate the tables published in our research. In order to use the correct experimental results, you can:

1. Change the `SWEEP_ID` in each analysis script to match the sweep ID in your own W&B workspace; or
2. Change the `WANDB_PROJECT` in `information_extraction/config.py` to `gijshendriksen/information_extraction`, to use the values from our own final experiments.

The `generate_results.sh` bash scripts runs all analyses automatically, and stores their outputs in a dedicated `outputs` directory.

#### Notes

1. The `analyse_webke` and `map_webke_to_original_docids` script requires the predictions from the WebKE system to be available in a `webke-predictions` artifact in W&B. This can be downloaded from [our own W&B workspace](https://wandb.ai/gijshendriksen/information_extraction/artifacts/predictions/webke-predictions).   
2. The `analyse_failures` script requires manual annotation. The `sample_failures` function samples a selection of failures from the model's outputs, and saves it in `failures.csv`. You must then copy this file to `failures_with_reasons.csv`, and add a `reason` column, in which you define the reason that each incorrect prediction was a failure. The `show_failure_types` function then takes this annotated file to summarize the failure types.