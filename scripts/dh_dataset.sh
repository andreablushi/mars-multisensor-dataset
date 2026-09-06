#!/usr/bin/env bash
set -euo pipefail

# The dataset is the one archive big enough to be worth asking for on its own,
# so it is not among the names dh_download.sh brings down together.
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

Brings one build of the dataset down and unpacks it under
data/building/dataset/<name>. With no name, the one configs/building_runner.yaml
is set to build.
TEXT
}

if [[ ${1-} == -h || ${1-} == --help ]]; then
    usage
    exit 0
fi

dest="data/building/dataset/$name"
# Everything lands beside the destination first, so a download that fails or is
# interrupted leaves what is already on disk untouched.
staged="$dest.incoming"
rm -rf "$staged"
mkdir -p "$staged"

echo "downloading $published-$name"
dhcli download -p "$project" artifact -n "$published-$name" -d "$staged"

packed="$(find "$staged" -maxdepth 1 -name '*.tar.gz' -print -quit)"
if [[ -z $packed ]]; then
    echo "nothing came down for \`$published-$name\`, leaving $dest as it was" >&2
    rm -rf "$staged"
    exit 1
fi

tar -xzf "$packed" -C "$staged" --strip-components=1
rm -f "$packed"

# The archive owns the directory it fills, so it replaces what is there rather
# than merging into it.
rm -rf "$dest"
mkdir -p "$dest"
cp -a "$staged"/. "$dest"/
rm -rf "$staged"
echo "unpacked into $dest"
