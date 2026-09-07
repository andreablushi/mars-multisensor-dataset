#!/usr/bin/env bash
set -euo pipefail

# The dataset is big enough to ask for on its own, so dh_download.sh leaves it
here="$(dirname "$0")"
platform="$here/../configs/digitalhub.yaml"
building="$here/../configs/building_runner.yaml"

project="$(sed -n 's/^project: *//p' "$platform")"
published="$(sed -n "/^publishes:/,/^[^ #]/{s/^  dataset: *//p;}" "$platform")"
# The build to bring down, this run's own unless one is named.
name="${1-$(sed -n 's/^name: *//p' "$building")}"

usage() {
    cat <<'TEXT'
usage: dh_dataset.sh [name]

Brings one build of the dataset down into data/building/dataset/<name>. With no
name, the one configs/building_runner.yaml is set to build. The dataset is
published a crop at a time, so this brings every one of them down; a training
run inside DigitalHub reads them from the store instead and never needs this.
TEXT
}

if [[ ${1-} == -h || ${1-} == --help ]]; then
    usage
    exit 0
fi

dest="data/building/dataset/$name"
# Everything lands beside the destination first, so a failure touches nothing
staged="$dest.incoming"
rm -rf "$staged"
mkdir -p "$staged"

echo "downloading $published-$name"
dhcli download -p "$project" artifact -n "$published-$name" -d "$staged"

if [[ -z $(ls -A "$staged" 2>/dev/null) ]]; then
    echo "nothing came down for \`$published-$name\`, leaving $dest as it was" >&2
    rm -rf "$staged"
    exit 1
fi

# The build owns the directory it fills, so it replaces rather than merges
rm -rf "$dest"
mkdir -p "$dest"
cp -a "$staged"/. "$dest"/
rm -rf "$staged"
echo "brought down into $dest"
