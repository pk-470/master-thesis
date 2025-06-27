# Master Thesis: Towards a Foundation Model for the Analysis of Environmental Audio Data Using the Transformer and Mamba Architectures

## Thesis text

The thesis text can be found in [Master_Thesis.pdf](Master_Thesis.pdf).

## Environment setup

See detailed setup instructions in [docs/SETUP.md](docs/SETUP.md).

## Experiments

Assuming the correct environment setup.

### Data

Download the waveforms:

```bash
bash scripts/download_data.sh [args]
```

Accepted args: e.g. `0-2` (inclusive range), `0 1` (exact), `all` (all data)

Alternatively, save the waveforms in a _dataset_ folder in the root directory with the following structure:

```text
.
|-- dataset
|   |-- waveforms
|   |   |-- GardenFiles23
|   |   |   |-- GardenFiles23_0
|   |   |   |-- GardenFiles23_1
|   |   |   |-- ...
```

If the waveforms are in a different location, define the following variable in a _.env_ file in the root directory:

```text
DATA_LOCATION=...
```

Convert the waveforms to spectrograms using the [scripts/convert_wav_to_spec.py](scripts/convert_wav_to_spec.py) script.

Generate train/validation/test splits using the [scripts/generate_splits.py](scripts/generate_splits.py) script or the [notebooks/prepare_data.ipynb](notebooks/prepare_data.ipynb) notebook.

Final dataset directory structure should look like:

```text
.
|-- dataset
|   |-- spectrograms
|   |   |-- one_channel
|   |   |   |-- GardenFiles23
|   |   |   |   |-- ...
|   |   |   |-- splits
|   |   |   |   |-- foundation
|   |   |   |   |   |-- test.csv
|   |   |   |   |   |-- train.csv
|   |   |   |   |   |-- val.csv
|   |   |   |   |-- ...
|   |   |-- two_channels
|   |   |   |-- GardenFiles23
|   |   |   |   |-- ...
|   |   |   |-- splits
|   |   |   |   |-- foundation
|   |   |   |   |   |-- test.csv
|   |   |   |   |   |-- train.csv
|   |   |   |   |   |-- val.csv
|   |   |   |   |-- ...
|   |-- waveforms
|   |   |-- GardenFiles23
|   |   |   |-- ...
|   |   |-- splits
|   |   |   |-- foundation
|   |   |   |   |-- test.csv
|   |   |   |   |-- train.csv
|   |   |   |   |-- val.csv
|   |   |   |-- ...
```

### Logging

To sync and visualise logs online using an existing [Weights & Biases](https://wandb.ai/site/) account, define the following variable in a _.env_ file in the root directory:

```text
WANDB_API_KEY=...
```

### Checkpoints

By default, checkpoints are stored in a _checkpoints_ folder in the root directory. To specify a different location, define the following variable in a _.env_ file in the root directory:

```text
CHECKPOINTS_LOCATION=...
```

### Training

Training configurations can be found in [src/spec_mamba/exp/cfg/](src/spec_mamba/exp/cfg/).

To run an experiment, i.e. train/test the model and log the results, use the [run](src/spec_mamba/exp/run.py) module as follows:

```bash
python -m spec_mamba.exp.run CONFIG_FILE_NAME [--seed=SEED]
```

Note that the configuration file name should not contain the _.py_ extension, e.g.:

```bash
python -m spec_mamba.exp.run cfg_aum --seed=42
```
