# Multi-Sensor Dataset Pipeline for Mars Tiles

This repository provides an automated analysis pipeline that splits Mars into equal-area tiles and gathers the observations of each one, using data from NASA's [Orbital Data Explorer (ODE)](https://ode.rsl.wustl.edu/mars/).

The pipeline measures what every instrument covers of each tile and filters the tiles according to specific coverage and temporal window criteria. Selected tiles and their corresponding multi-sensor observations are then processed into standardized data, ready for training and evaluation of machine learning models.

## Repository layout

What both datasets share lives in `common`, and what only one of them does lives
in its own package. The configs follow the same split.

```
src/common/       the analysis and the build both datasets run through
src/training/     the share of the kept tiles the training build draws
src/evaluation/   the labels, the balanced draw, and what its notebook draws
configs/common/   analysis.yaml, building.yaml, digitalhub.yaml
configs/training/ building.yaml
configs/evaluation/ analysis.yaml, building.yaml
scripts/          analysis_pipeline.py, training_pipeline.py, evaluation_pipeline.py
```

The scripts stay flat, since the scripts root is on the import path and a
`training` or `evaluation` directory there would shadow the packages of the same name.

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
for, is in `configs/common/digitalhub.yaml`.
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
Every other choice comes from `configs/common/analysis.yaml`, so the same files describe what was run
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
`configs/common/digitalhub.yaml`, so nothing here can drift from what the pipeline
publishes.

| Name | What it holds | Where it lands |
| --- | --- | --- |
| `coverage` | the coverage measurements | `data/analysis/coverage/` |
| `summary` | one row per tile and instrument set | `data/analysis/coverage/` |
| `metadata` | the ODE records behind the measurements | `data/analysis/metadata/` |
| `selection` | the tiles and observations the filter keeps | `data/analysis/selection/` |
| `stats` | what the filter left of the dataset | `data/analysis/stats/` |
| `labels` | every labelled tile, the drawn ones marked | `data/evaluation/labels/` |
| `dataset` | the cropped observations and their index | `data/building/dataset/` |

A build is big enough to be asked for on its own, by its name:

```bash
./scripts/dh_dataset.sh             # the training build
./scripts/dh_dataset.sh evaluation  # the evaluation build
```

## Building the training dataset

Starting from the previous selection, the pipeline builds a dataset where a sample is defined as a tile and its corresponding multi-sensor observations.

```bash
uv run python scripts/training_pipeline.py          # here
uv run --group digitalhub python scripts/training_pipeline.py --dh
```

What the training build draws is described in `configs/training/building.yaml`,
and how every build runs in `configs/common/building.yaml`. The tiles the
evaluation set drew are held out of it, so the evaluation labels have to be
written first, and a tile an earlier build held that it no longer draws leaves
its index.

## Building the evaluation dataset

The evaluation set tests whether a model tells geology apart: every tile of one
class should lie near the others of it and far from every other class. Its tiles
are drawn out of the ones the selection kept, on the same 32 km grid and under
the same filter, and labelled by the feature catalogue ODE publishes, the IAU
nomenclature, so no source beyond ODE is read.

```bash
uv run python scripts/evaluation_pipeline.py --only-labels   # label and draw
uv run python scripts/evaluation_pipeline.py                 # and build
uv run --group digitalhub python scripts/evaluation_pipeline.py --dh
```

Every class is set in `configs/evaluation/analysis.yaml`. A texture class holds a
tile lying in the middle of one of its features, since any patch of it shows
what it is. An object class, the crater, holds a tile a crater of 8 to 16 km lies
in whole, so every one sits in its tile at a similar scale. A tile two classes
claim is left out, and so is a texture tile any crater reaches into. The draw
then takes as many tiles of every class as the scarcest holds, one feature at a
time in turn, so no single feature fills its class.

| Class | Read from |
| --- | --- |
| `crater` | Crater, 8 to 16 km, whole in the tile |
| `chaos` | Chaos |
| `dune_field` | Unda |
| `fossae` | Fossa |
| `labyrinthus` | Labyrinthus |
| `polar_layered_deposits` | Planum Boreum, Planum Australe |
| `shield_volcano` | Olympus, Ascraeus, Pavonis, Arsia, Alba and Elysium Mons |
| `ice_rich_plains` | Arcadia and Utopia Planitia, 38 to 50 N |
| `ice_poor_plains` | Amazonis and Elysium Planitia, 30 S to 30 N |

The last two are the SHARAD pair: plains alike to the eye, one holding buried
ice where SHARAD finds it, the other too warm to keep any. They are told apart
under the surface echo of the radargram, which the evaluation notebook sets side
by side. The drawn labels are written beside the crops, as `labels.parquet`.

## Dataset structure

The dataset is entirely self-contained within a single directory, consisting of individual observation crops and a master index mapping each file. Published as separate objects rather than a single compressed archive, each crop is keyed to its index path. During training, jobs read the index and fetch only the sampled crops directly from storage, avoiding full dataset downloads in the DigitalHub.

```
dataset/
  dataset.json          what this dataset is: format version, when, from what
  tiles.parquet         one row per tile: where it is, and what was kept of it
  observations.parquet  one row per crop: its shape, its ground, its statistics
  labels.parquet        the evaluation build alone: the class of every tile
  <band>/<column>/<instrument>/<identifier>.npz
```

Each crop is stored as an .npz file containing spatial data arrays and an embedded meta JSON object. The metadata specifies axis names, indicates which axes correspond to ground coordinates, defines the cropped tile's location, and records the original publication labels for every product.

## Notebooks

`notebooks/training_qualitative.ipynb` reads one tile at a time, whole. Type a latitude
and longitude, confirm, and the cells below fill themselves in for the tile
holding that point. An instrument that reached none
of it is still drawn, at zero, so a missing line always means something.

`notebooks/evaluation_quantitative.ipynb` reads the evaluation set: every class,
where its tiles lie, what the instruments land on them, and one built tile of
two classes set side by side.

`notebooks/training_quantitative.ipynb` reads what the filter made of every measured
tile rather than of a sample of them, and maps every tile of Mars, green where
the filter kept it and red where it did not. It reads back what the pipeline
published and builds no artifact of its own.
