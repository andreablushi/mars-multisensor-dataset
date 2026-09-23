# Multi-Sensor Dataset Pipeline for Mars Tiles

A pipeline that splits Mars into equal-area tiles, gathers the observations of each one from NASA's [Orbital Data Explorer (ODE)](https://ode.rsl.wustl.edu/mars/), and builds them into a training and an evaluation dataset.

## Setup

```bash
uv sync                                     # environment
uv run ruff check . && uv run ruff format . # lint, over the whole repo
```

CTX is calibrated and projected by [USGS ISIS](https://github.com/DOI-USGS/ISIS3), which a local build needs beside the environment, with `ISISROOT` and `ISISDATA` set. `scripts/dhub/isis.py` holds the version and the data it downloads, and runs the same install on DigitalHub.

## Running it

Run the three scripts in this order, each here by default or on DigitalHub with `--dh`:

```bash
uv run python scripts/analysis_pipeline.py   # coverage, ancillary, selection, stats and labels
uv run python scripts/build_training.py      # the training dataset
uv run python scripts/build_evaluation.py    # the evaluation dataset
```

Both builds read what the analysis wrote, so the analysis has to have run first.

| Flag | Script | What it does |
| --- | --- | --- |
| `--only-stats` | `analysis_pipeline.py` | skip the download and the measurement, and select, label and read the stats from what is already on disk |
| `--force` | all | redo finished work rather than skip it |
| `--dh` | all | submit to DigitalHub instead of running here |
| `--ref` | all | with `--dh`, the branch, tag, or commit the platform clones |

## Running it on DigitalHub

[DigitalHub](https://scc-digitalhub.github.io/docs/0.15/) clones a pushed commit, so every change has to be on the branch you name with `--ref` before it can run.

```bash
uv sync --group digitalhub
dhcli register <your-digitalhub-core-endpoint>
dhcli login                                    # opens a browser tab

cp .env.example .env                           # once, then fill it in

uv run --group digitalhub python scripts/analysis_pipeline.py --dh
uv run --group digitalhub python scripts/build_training.py --dh
uv run --group digitalhub python scripts/build_evaluation.py --dh
```

## Downloading what it published

```bash
chmod +x scripts/dh_download.sh scripts/dh_dataset.sh   # once
./scripts/dh_download.sh                 # every analysis archive
./scripts/dh_download.sh selection stats # only some of them
```

| Name | Where it lands |
| --- | --- |
| `coverage`, `summary` | `data/analysis/coverage/` |
| `metadata` | `data/analysis/metadata/` |
| `selection` | `data/analysis/selection/` |
| `stats` | `data/analysis/stats/` |
| `labels` | `data/analysis/labels/` |
| a dataset | `data/building/dataset/<name>/` |

## Notebooks

```bash
uv sync --group notebook
uv run --group notebook jupyter lab
```

- `notebooks/training_qualitative.ipynb`: one tile at a time, picked by latitude and longitude.
- `notebooks/training_quantitative.ipynb`: what the selection kept of every tile.
- `notebooks/evaluation_quantitative.ipynb`: the evaluation classes, where they lie, and what the instruments land on them.
