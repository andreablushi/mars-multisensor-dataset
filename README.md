# Multi-Sensor Dataset Pipeline for Mars Tiles

This repository provides an automated analysis pipeline that splits Mars into equal-area tiles and gathers the observations of each one, using data from NASA's [Orbital Data Explorer (ODE)](https://ode.rsl.wustl.edu/mars/).

The pipeline measures what every instrument covers of each tile and filters the tiles according to specific coverage and temporal window criteria. Selected tiles and their corresponding multi-sensor observations are then processed into standardized data, ready for training and evaluation of machine learning models.

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
for, is in `configs/digitalhub.yaml`.
It's important to fill in the `.env` file with the correct values.
Refer to the `.env.example` file for the required variables and their descriptions.

### Running it locally

It's possible to run the pipeline locally, by simply omitting the `--dh` flag.
The local run will use the same configs and produce the same outputs, but it will not be versioned or managed by DigitalHub.


## Running the analysis

One entry point runs every stage, here by default:

```bash
uv run python scripts/analysis_pipeline.py
```

It downloads the ODE metadata of every tile group, measures the coverage of
every tile, searches each for its best window, and writes what the filter keeps.
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
| `summary` | one row per tile and instrument set | `data/analysis/coverage/` |
| `metadata` | the ODE records behind the measurements | `data/analysis/metadata/` |
| `selection` | the tiles and observations the filter keeps | `data/analysis/selection/` |
| `stats` | what the filter left of the dataset | `data/analysis/stats/` |
| `dataset` | the cropped observations and their index | `data/building/dataset/` |

## Building the dataset

Starting from the previous selection, the pipeline builds a dataset where a sample is defined as a tile and its corresponding multi-sensor observations.

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
  tiles.parquet         one row per tile: where it is, and what was kept of it
  observations.parquet  one row per crop: its shape, its ground, its statistics
  <band>/<column>/<instrument>/<identifier>.npz
```

Each crop is stored as an .npz file containing spatial data arrays and an embedded meta JSON object. The metadata specifies axis names, indicates which axes correspond to ground coordinates, defines the cropped tile's location, and records the original publication labels for every product.

## Notebooks

`notebooks/qualitative.ipynb` reads one tile at a time, whole. Type a latitude
and longitude, confirm, and the cells below fill themselves in for the tile
holding that point. An instrument that reached none
of it is still drawn, at zero, so a missing line always means something.

`notebooks/quantitative.ipynb` reads what the filter made of every measured
tile rather than of a sample of them, and maps every tile of Mars, green where
the filter kept it and red where it did not. It reads back what the pipeline
published and builds no artifact of its own.
