# Multi-Sensor Dataset Pipeline for Mars Tiles

A pipeline that splits Mars into equal-area tiles, gathers the observations of each one from NASA's [Orbital Data Explorer (ODE)](https://ode.rsl.wustl.edu/mars/), and builds them into a training and an evaluation dataset.

## Setup

```bash
uv sync                                     # environment
uv run ruff check . && uv run ruff format . # lint, over the whole repo
```

CTX is calibrated and projected by [USGS ISIS](https://github.com/DOI-USGS/ISIS3), which a local build needs beside the environment, with `ISISROOT` and `ISISDATA` set. `src/mars_multisensor_dataset/building/preprocessing/ctx/isis.py` holds the version and the data it downloads, and installs it when a DigitalHub job starts.

## Running it

Run the three scripts in this order, each here by default or on DigitalHub with `--dh`:

```bash
uv run python scripts/analysis_pipeline.py   # coverage, ancillary, selection, stats, labels and ice
uv run python scripts/build_training.py      # the training dataset
uv run python scripts/build_evaluation.py    # the evaluation dataset
```

Both builds read what the analysis wrote, so the analysis has to have run first.

## Dataset versions

`configs/version.yaml` names the dataset version every run reads and writes, `halfplanet` by default and `seasonal` for the older build. The selection, stats, labels, verdicts and both builds of a version live under `data/<version>/`, and on DigitalHub under `artifacts/<version>/` as `<version>-selection`, `<version>-dataset-training` and so on. Coverage, metadata, the SWIM maps and the download caches do not depend on the selection, so every version shares them under `data/analysis/` and `data/building/`.

| Flag | Script | What it does |
| --- | --- | --- |
| `--only-stats` | `analysis_pipeline.py` | skip the download and the measurement, and select, label and read the stats from what is already on disk |
| `--only-lavaflow` | `analysis_pipeline.py` | list the lava flow crops alone, here, from the coverage already on disk |
| `--force` | all | redo finished work rather than skip it |
| `--dh` | all | submit to DigitalHub instead of running here |
| `--ref` | all | with `--dh`, the branch, tag, or commit the platform clones |

## Running it on DigitalHub

[DigitalHub](https://scc-digitalhub.github.io/docs/0.15/) clones a pushed commit, so every change has to be on the branch you name with `--ref` before it can run.

```bash
uv sync --group digitalhub
dhcli register <your-digitalhub-core-endpoint>
dhcli login                                    # opens a browser tab

uv run --group digitalhub python scripts/analysis_pipeline.py --dh
uv run --group digitalhub python scripts/build_training.py --dh
uv run --group digitalhub python scripts/build_evaluation.py --dh
```

A job installs `requirements.txt` when it starts. Export it again whenever `uv.lock` changes, then commit and push it:

```bash
uv export --no-hashes --no-dev --group digitalhub --no-emit-project -o requirements.txt
```

## Downloading what it published

```bash
chmod +x scripts/dh_download.sh   # once
./scripts/dh_download.sh                 # every analysis archive
./scripts/dh_download.sh selection stats # only some of them
```

| Name | Where it lands |
| --- | --- |
| `coverage` | `data/analysis/coverage/` |
| `metadata` | `data/analysis/metadata/` |
| `swim` | `data/analysis/swim/` |
| `selection` | `data/<version>/analysis/selection/` |
| `stats` | `data/<version>/analysis/stats/` |
| `labels` | `data/<version>/analysis/labels/` |
| `verdicts` | `data/<version>/analysis/` |
| `manifest` | `data/<version>/dataset/training/` |
| `lavaflow` | `data/<version>/dataset/lavaflow/`, the reviewed crops, only when named |
| `lavaflow_candidates` | `data/analysis/lavaflow_candidates/`, every candidate crop, only when named |

## Notebooks

```bash
uv sync --group notebook
uv run --group notebook jupyter lab
```

- `notebooks/training_qualitative.ipynb`: one tile at a time, picked by latitude and longitude.
- `notebooks/training_quantitative.ipynb`: what the selection kept of every tile.
- `notebooks/evaluation_quantitative.ipynb`: the evaluation classes, where they lie, and what the instruments land on them, then the SWIM ice under every tile.
- `notebooks/annotation_nomenclature.ipynb`: every labelled tile of the evaluation classes on the THEMIS mosaic, in draw order, to accept or reject.
- `notebooks/annotation_lavaflow.ipynb`: every lava flow crop beside its cluttergram, to mark as reflector, none or unsure.

## Uploading the annotation

The nomenclature annotation notebook writes its verdicts to `data/<version>/analysis/verdicts.json`. Upload them, then select again so the draw skips the rejected tiles and training holds them out:

```bash
uv run --group digitalhub python scripts/dh_upload.py
uv run --group digitalhub python scripts/analysis_pipeline.py --only-stats --dh
```

## The lava flow build

A third build holds SHARAD alone, for retrieving buried lava flow reflectors. `ground_truth.lavaflow` in `configs/analysis.yaml` lists the regions where papers report them and the tracks each paper names. The analysis lists one crop per track and tile of a region that the selection's own admit takes (`admits.SHARAD` traces on the tile), kept by the selection or not, and flags the tracks the paper names as `named`. A crop the build cannot fill with one whole patch is emptied, as in every build.

```bash
uv run python scripts/analysis_pipeline.py --only-lavaflow   # list the crops
uv run python scripts/build_lavaflow.py --annotation         # build every candidate, or --dh
./scripts/dh_download.sh lavaflow_candidates                 # after a --dh build
```

Then review the crops in `notebooks/annotation_lavaflow.ipynb`, which writes `data/<version>/analysis/verdicts_lavaflow.json`; `dh_upload.py` uploads this file together with `verdicts.json`. `build_lavaflow.py` without `--annotation` then builds, from nothing, only the crops reviewed as reflector or none into `data/<version>/dataset/lavaflow`, with `lavaflow_labels.parquet` holding each crop's verdict and `named` flag. It deletes what that build held before, on disk or published, so an unsure crop is never fetched; run it with `--dh`, since it downloads every crop again.
