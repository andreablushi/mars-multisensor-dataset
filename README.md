# Mars Multi-Sensor Observations Dataset

A multi-sensor build dataset pipeline of Mars geological features. One sample is a single named
landform seen by three instruments inside one shared time window: CTX visible
imagery, CRISM multispectral cubes, and SHARAD radar sounding, each cropped to
that landform's extent.

## Development commands

```bash
uv sync                                     # environment
uv run ruff check . && uv run ruff format . # lint, over the whole repo
git config core.hooksPath .githooks         # once per clone, blocks unlinted pushes
```

## Running the analysis

One entry point runs every stage, here by default:

```bash
uv run python scripts/analysis_pipeline.py
```

It downloads the ODE metadata still missing, measures the coverage of every
feature, searches each for its best window, and writes what the filter keeps.
Every other choice comes from `configs/`, so the same files describe what was run
and what to run again.

| Flag | What it does |
| --- | --- |
| `--only-stats` | skip the download and the measurement, and select from what is already on disk |
| `--force` | redo finished work rather than skip it: download again and measure again |
| `--dh` | submit to DigitalHub instead of running here |
| `--ref` | with `--dh`, the branch, tag, or commit the platform clones |

## Running it on DigitalHub

[DigitalHub](https://scc-digitalhub.github.io/docs/0.15/) runs the same entry
point on a cluster and keeps what it produced as versioned entities. It clones a
pushed commit, so every change has to be on the branch you name with `--ref`
before it can run.

```bash
uv sync --group digitalhub
dhcli register <your-digitalhub-core-endpoint>
dhcli login                                    # opens a browser tab

uv run --group digitalhub python scripts/analysis_pipeline.py --dh
uv run --group digitalhub python scripts/analysis_pipeline.py --dh --only-stats
```

The first form runs every stage in one job and publishes all of it: the
measurements, the catalogue index, the records behind them, and the selection
and stats the filter left. The second reads a published measurement back and
publishes only the selection and the stats, so re-running an edited filter costs
nothing but the search.

Everything a submission needs, from the project name to the memory a job asks
for, is in `configs/digitalhub.yaml`. The pip requirements are taken straight
from `pyproject.toml`, so the image always matches this repository.

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
| `catalog` | the ODE feature and instrument set lists | `data/_catalog/` |
| `metadata` | the ODE records behind the measurements | `data/analysis/metadata/` |
| `selection` | the features and observations the filter keeps | `data/analysis/selection/` |
| `stats` | what the filter left of the dataset | `data/analysis/stats/` |
| `dataset` | the cropped observations and their index | `data/building/dataset/` |

## Building the dataset

```bash
uv run python scripts/building_pipeline.py          # here
uv run --group digitalhub python scripts/building_pipeline.py --dh
```

`configs/building_runner.yaml` says how much to build. `share` is what fraction
of the features the selection kept to build, drawn evenly across their classes,
and `name` is what that build is called: it is the directory it is written in
and the name it is published under, so a half build and a whole one sit side by
side. A feature is built whole, with every observation the selection left it.
The same seed and a larger share gives a superset, so a small build is always
part of the larger one.

`ready` holds the downloads to the room they were given, so they cannot race
ahead of the builds that consume them. A product is deleted once every feature
that wanted it has been cut, so a build needs room for what it holds at once and
never for everything it ever fetched. Keep `ready` well above `workers`, or the
build pool starves waiting for products.

## Using the dataset

The dataset is one directory: a crop per observation of a feature, and beside
them the index that says what each is. Nothing outside it is needed to read it.

```
dataset/
  dataset.json          what this dataset is: format version, when, from what
  features.parquet      one row per feature: where it is, and what was kept of it
  observations.parquet  one row per crop: its shape, its ground, its statistics
  <class>/<feature>/<instrument>/<identifier>.npz
```

Each crop is a single `.npz`. Beside its arrays it carries a `meta` entry, a
JSON object saying what every array's axes are called, which of them are ground,
where the feature it was cut to sits, and the label every product it was
published as was written with.

```python
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

root = Path("data/building/dataset")
rows = pq.read_table(root / "observations.parquet").to_pylist()

held = np.load(root / rows[0]["path"], allow_pickle=False)
meta = json.loads(str(held["meta"]))
values = held[meta["measurement"]]  # what the instrument measured
```

The index is read on its own, so a count, a filter or a split opens no array at
all. A split is drawn over features and never over observations, since one
feature is seen in many observations and splitting those would put the same
ground on both sides of it.

Every crop holds its measurement placed by `north` and `east`, in degrees from
the feature's own centre. Nothing in an array says where on Mars its feature is,
so adding the centre back is what turns a placement into a coordinate:

```python
latitude = meta["centre_lat"] + held["north"]
longitude = meta["centre_lon"] + held["east"]
```

An axis named in `meta["ground"]` is ground; the others are the instrument's own
and are sampled in their own unit, which `meta["axes"]` names. A grid places one
axis each and a swath or a track places every sample, which is what `separable`
says. Two masks narrow what is a measurement, and each is written only when it
excludes something, so an absent one means every sample is kept:

```python
kept = np.ones(
    [
        values.shape[meta["dims"][meta["measurement"]].index(one)]
        for one in meta["ground"]
    ],
    bool,
)
for name in ("inside", "valid"):  # in the box, and measured
    if name in held.files:
        kept &= held[name]
```

`inside` is unset for a map raster, which meets a feature's box in a rectangle,
and set for a swath or a track, which does not. `valid` is unset where every
sample is a measurement.

Each row of the index carries the statistics of its own crop, over the samples
those two masks keep. They pool exactly, since every row says how many values it
was measured over, and an average of per-crop means or standard deviations would
not. Pool them per instrument, since each measures its own quantity:

```python
held_rows = [one for one in rows if one["instrument"] == "CRISM"]
n = sum(one["valid_count"] for one in held_rows)
mean = sum(one["value_mean"] * one["valid_count"] for one in held_rows) / n
var = (
    sum(
        one["valid_count"] * (one["value_std"] ** 2 + (one["value_mean"] - mean) ** 2)
        for one in held_rows
    )
    / n
)
```

## Notebooks

`notebooks/qualitative.ipynb` reads one feature at a time, whole. Pick a feature,
confirm, and the cells below fill themselves in. An instrument that reached none
of it is still drawn, at zero, so a missing line always means something.

`notebooks/quantitative.ipynb` reads what the filter made of every measured
feature rather than of a sample of them. It reads back what the pipeline
published and builds no artifact of its own.

## Configuration

```
configs/
  analysis_runner.yaml  # What a run downloads and measures, and on how many workers
  window_filter.yaml    # What a window has to hold for a feature to earn a place
  building_runner.yaml  # How much of the dataset to build, and on how many workers
  digitalhub.yaml       # What a submitted run is given, and what it publishes
```

Each file is read as written and no setting is checked: the run that reads it is
the check. What changes from one run to the next is a flag instead.

## Structure

```
scripts/
  analysis_pipeline.py  # Measures what the archives cover, and selects from it
  building_pipeline.py  # Builds the dataset the selection asks for
  dh_download.sh        # Brings the analysis entities back down
  dh_dataset.sh         # Brings one build of the dataset back down
  dhub/                 # Only what a submitted run needs
    configs.py          # Reads configs/digitalhub.yaml
    archives.py         # Packs what is published, unpacks what is read back
    submit.py           # Registers a version of a function, and starts the job
notebooks/              # The two notebooks that read the results
src/
  utils/                # What both halves use
    ode/                # The ODE client, its settings, its errors
    disk/               # Project paths, atomic writes, slugs, parquet
    geometry/           # Mars, its longitudes and the local projection
  analysis/             # What the archives cover, and what the notebooks read
    console.py          # Progress bars and totals
    planner.py          # What each half has left to do
    runner.py           # Holds the survey's plan and its pools
    models/             # Features, instruments, settings, jobs
    metadata/           # Asking ODE for records, and reading them back
    coverage/           # The coverage measurement, its geometry and its artifacts
    selector/           # The best time window search, under one filter
    stats/              # What the filter left, measured over one feature or all
    visualization/      # What the notebooks draw
  building/             # What a chosen observation is turned into
    configs/            # What each instrument is, read by every stage
    common/             # Naming, the product cache, the PDS formats
    download/           # Bringing down what the selection kept
    preprocessing/      # One product read, cut to a feature, and written down
      common/           # What every instrument's crop shares
      crism/ ctx/       # What each instrument reads and cuts of its own
      mola/ sharad/
    metadata/           # What the dataset, its features and its crops are
    models/             # The frame, the jobs, the settings
    dispatcher.py       # What each instrument does at every stage of a build
    planner.py          # What a build has to fetch and cut
    runner.py           # Holds the build's pools
    console.py          # Progress and totals
data/                   # Laid out as src is, each half owning what it writes
  _catalog/             # Cached ODE catalogs, read by both halves
  analysis/
    metadata/           # Raw ODE records
    coverage/
      features/         # Per-feature coverage measurements
      summary.parquet   # Every feature's summary rows together
    selection/          # The features and observations the filter keeps
    stats/              # What the filter left of the dataset
  building/
    dataset/            # The built dataset: crops, and the index over them
    preprocessing/      # One directory per instrument, holding its products
```
