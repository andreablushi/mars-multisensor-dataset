#!/usr/bin/env bash
set -euo pipefail

# Every name is read from the file the runs are settled from, so nothing drifts
config="$(dirname "$0")/../configs/digitalhub.yaml"
project="$(sed -n 's/^project: *//p' "$config")"

published() {
    sed -n "/^publishes:/,/^[^ #]/{s/^  $1: *//p;}" "$config"
}

download_one() {
    local name="$1"
    local dest="$2"
    local shares="${3-}"

    # Everything lands beside the destination first, so a failure touches nothing
    local staged="$dest.incoming"
    rm -rf "$staged"
    mkdir -p "$staged"

    dhcli download -p "$project" artifact -n "$name" -d "$staged"

    # An archive comes down packed, the summary as the one file it is
    local packed
    packed="$(find "$staged" -maxdepth 1 -name '*.tar.gz' -print -quit)"
    if [[ -n $packed ]]; then
        tar -xzf "$packed" -C "$staged" --strip-components=1
        rm -f "$packed"
    fi

    # Nothing on disk is touched unless there is something to put in its place
    if [[ -z $(ls -A "$staged" 2>/dev/null) ]]; then
        echo "nothing came down for \`$name\`, leaving $dest as it was" >&2
        rm -rf "$staged"
        return 1
    fi

    # An archive owns the directory it fills, though the summary shares one
    [[ -n $shares ]] || rm -rf "$dest"
    mkdir -p "$dest"
    cp -a "$staged"/. "$dest"/
    rm -rf "$staged"
}

# dh_dataset.sh takes the two above for the one archive this leaves it, so
# everything below here belongs to a direct run alone.
[[ ${BASH_SOURCE[0]} == "$0" ]] || return 0

usage() {
    cat <<'EOF'
usage: dh_download.sh [name ...]

Brings down what the pipelines published and unpacks it under data/.
With no name, every one of them comes down.

  coverage     the coverage measurements       -> data/analysis/coverage
  catalog      the ODE feature and set lists   -> data/_catalog
  metadata     the ODE records behind them     -> data/analysis/metadata
  selection    the features and looks kept    -> data/analysis/selection
  stats        what the filter left of it     -> data/analysis/stats
  summary      one row per feature and set     -> data/analysis/coverage
EOF
}

if [[ ${1-} == -h || ${1-} == --help ]]; then
    usage
    exit 0
fi

names=("$@")
if [[ ${#names[@]} -eq 0 ]]; then
    names=(coverage catalog metadata selection stats summary)
fi

for name in "${names[@]}"; do
    case "$name" in
        coverage) download_one "$(published coverage)" data/analysis/coverage ;;
        catalog) download_one "$(published catalog)" data/_catalog ;;
        metadata) download_one "$(published metadata)" data/analysis/metadata ;;
        selection) download_one "$(published selection)" data/analysis/selection ;;
        stats) download_one "$(published stats)" data/analysis/stats ;;
        summary) download_one "$(published summary)" data/analysis/coverage shares ;;
        *)
            echo "nothing is published under \`$name\`" >&2
            usage >&2
            exit 1
            ;;
    esac
done
