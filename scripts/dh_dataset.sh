#!/usr/bin/env bash
set -euo pipefail

# The dataset is big enough to ask for on its own, so dh_download.sh leaves it
here="$(dirname "$0")"
source "$here/dh_download.sh"
building="$here/../configs/building.yaml"
# The build to bring down, this run's own unless one is named.
name="${1-$(sed -n 's/^name: *//p' "$building")}"

usage() {
    cat <<'TEXT'
usage: dh_dataset.sh [name]

Brings one build of the dataset down into data/building/dataset/<name>. With no
name, the one configs/building.yaml is set to build. The dataset is
published a crop at a time, so this brings every one of them down; a training
run inside DigitalHub reads them from the store instead and never needs this.
TEXT
}

if [[ ${1-} == -h || ${1-} == --help ]]; then
    usage
    exit 0
fi

dest="data/building/dataset/$name"
echo "downloading $(published dataset)-$name"
download_one "$(published dataset)-$name" "$dest"
echo "brought down into $dest"
