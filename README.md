# Multi-Sensor Dataset Pipeline for Mars Geological Features

This repository provides an automated analysis pipeline to process and extract geological features and observations on Mars using data from NASA's [Orbital Data Explorer (ODE)](https://ode.rsl.wustl.edu/mars/).

The pipeline checks feature availability and filters geological targets according to specific coverage and temporal window criteria. Selected features and their corresponding multi-sensor observations are then processed into standardized data, ready for training and evaluation of machine learning models.

## Development commands

```bash
uv sync                                     # environment
uv run ruff check . && uv run ruff format . # lint, over the whole repo
```

## Running it on DigitalHub

This repository was built to run on [DigitalHub](https://scc-digitalhub.github.io/docs/0.15/), a platform that manages and executes data processing pipelines in a distributed environment. DigitalHub runs the same entry point on a cluster and keeps what it produced as versioned entities. It clones a pushed commit, so every change has to be on the branch you name with `--ref` before it can run.

```bash
uv sync --group digitalhub
dhcli register <your-digitalhub-core-endpoint>
dhcli login                                    # opens a browser tab

cp .env.example .env                           # once, then fill it in

uv run --group digitalhub python scripts/analysis_pipeline.py --dh
uv run --group digitalhub python scripts/analysis_pipeline.py --dh --only-stats
```

Everything a submission needs, from the project name to the memory a job asks
for, is in `configs/digitalhub.yaml`. The pip requirements are taken straight
from `pyproject.toml`, so the image always matches this repository.

A run outlasts the credentials it is started with: those lapse after some six
hours, and a build that is still going then can publish nothing. So a job mints
its own instead. It presents a personal access token, which the platform holds
as a secret named `DHCORE_PERSONAL_ACCESS_TOKEN` and hands the job under that
name; create one from the console, under your username, then Configuration, then
Personal Access Tokens. The token names neither who issues credentials nor who
asks for them, so `.env` carries those two, and a submission forwards them to
the job. It is not committed, `.env.example` says what it holds, and a
submission stops before it starts anything if either is missing.

It's possible to run the pipeline locally, by simply omitting the `--dh` flag. The local run will use the same configs and produce the same outputs, but it will not be versioned or managed by DigitalHub.


## Running the analysis

One entry point runs every stage, here by default:

```bash
uv run python scripts/analysis_pipeline.py
```

It downloads the ODE metadata, measures the coverage of every
feature, searches each for its best window, and writes what the filter keeps.
Every other choice comes from `configs/`, so the same files describe what was run
and what to run again.

| Flag | What it does |
| --- | --- |
| `--only-stats` | skip the download and the measurement, and select from what is already on disk |
| `--force` | redo finished work rather than skip it: download again and measure again |
| `--dh` | submit to DigitalHub instead of running locally |
| `--ref` | with `--dh`, the branch, tag, or commit the platform clones |

## Downloading what it published

```bash
chmod +x scripts/dh_download.sh   # once, to make it executable
./scripts/dh_download.sh          # everything
./scripts/dh_download.sh selection stats
```

The names, and the project they come from, are read out of
`configs/digitalhub.yaml`, so nothing here can drift from what the pipeline
publishes.

| Name | What it holds | Where it lands |
| --- | --- | --- |
| `coverage` | the coverage measurements | `data/analysis/coverage/` |
| `summary` | one row per feature and instrument set | `data/analysis/coverage/` |
| `catalog` | the ODE feature set list | `data/_catalog/` |
| `metadata` | the ODE records behind the measurements | `data/analysis/metadata/` |
| `selection` | the features and observations the filter keeps | `data/analysis/selection/` |
| `stats` | what the filter left of the dataset | `data/analysis/stats/` |
| `dataset` | the cropped observations and their index | `data/building/dataset/` |

## Building the dataset

Starting from the previous selection, the pipeline builds a dataset where a sample is defined as a geological feature and its corresponding multi-sensor observations.

```bash
uv run python scripts/building_pipeline.py          # here
uv run --group digitalhub python scripts/building_pipeline.py --dh
```

Every run choices of the dataset to build are described in `configs/building.yaml`.

### Dataset structure

The dataset is entirely self-contained within a single directory, consisting of individual observation crops and a master index mapping each file. Published as separate objects rather than a single compressed archive, each crop is keyed to its index path. During training, jobs read the index and fetch only the sampled crops directly from storage, avoiding full dataset downloads in the DigitalHub.

```
dataset/
  dataset.json          what this dataset is: format version, when, from what
  features.parquet      one row per feature: where it is, and what was kept of it
  observations.parquet  one row per crop: its shape, its ground, its statistics
  <class>/<feature>/<instrument>/<identifier>.npz
```

Each crop is stored as an .npz file containing spatial data arrays and an embedded meta JSON object. The metadata specifies axis names, indicates which axes correspond to ground coordinates, defines the cropped feature's location, and records the original publication labels for every product.

## Notebooks

`notebooks/qualitative.ipynb` reads one feature at a time, whole. Pick a feature,
confirm, and the cells below fill themselves in. An instrument that reached none
of it is still drawn, at zero, so a missing line always means something.

`notebooks/quantitative.ipynb` reads what the filter made of every measured
feature rather than of a sample of them. It reads back what the pipeline
published and builds no artifact of its own.

`notebooks/crism_preprocessing.ipynb` takes one CRISM observation apart, a
correction at a time, drawing the cube after each one. It brings that
observation down itself, so it waits on no pipeline and on nothing already
on disk.

## Configuration

```
configs/
  analysis.yaml         # What a run downloads and measures, and what a window must hold
  building.yaml         # How much of the dataset to build, and what to call it
  digitalhub.yaml       # What a submitted run is given, and what it publishes
```

Each file is read as written and no setting is checked: the run that reads it is
the check. What changes from one run to the next is a flag instead.
