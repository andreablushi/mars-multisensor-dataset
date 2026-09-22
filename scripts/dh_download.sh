#!/usr/bin/env bash
set -euo pipefail

# The project and the version are read from the files the runs are settled from
configs="$(dirname "$0")/../configs"
project="$(sed -n 's/^project: *//p' "$configs/digitalhub.yaml")"
version="$(sed -n 's/^version: *//p' "$configs/version.yaml")"

download_one() {
    local name="$1"
    local dest="$2"
    local shares="${3-}"

    # Everything lands beside the destination first, so a failure touches nothing
    local staged="$dest.incoming"
    rm -rf "$staged"
    mkdir -p "$staged"

    dhcli download -p "$project" artifact -n "$name" -d "$staged"

    # Nothing on disk is touched unless there is something to put in its place
    if [[ -z $(ls -A "$staged" 2>/dev/null) ]]; then
        echo "nothing came down for \`$name\`, leaving $dest as it was" >&2
        rm -rf "$staged"
        return 1
    fi

    # An archive owns the directory it fills, though the verdicts share one
    [[ -n $shares ]] || rm -rf "$dest"
    mkdir -p "$dest"
    cp -a "$staged"/. "$dest"/
    rm -rf "$staged"
}

# Everything below runs only when this is executed rather than sourced.
[[ ${BASH_SOURCE[0]} == "$0" ]] || return 0

usage() {
    cat <<'EOF'
usage: dh_download.sh [name ...]

Brings down what the pipelines published and unpacks it under data/, the
version's own under data/<version>, the version set in configs/version.yaml.
With no name, every one of them comes down.

  coverage     the coverage measurements       -> data/analysis/coverage
  metadata     the ODE records behind them     -> data/analysis/metadata
  swim         the SWIM maps and ice labels    -> data/analysis/swim
  selection    the tiles and looks kept       -> data/<version>/analysis/selection
  stats        what the filter left of it     -> data/<version>/analysis/stats
  labels       the evaluation labels           -> data/<version>/analysis/labels
  verdicts     the review of the drawn tiles   -> data/<version>/analysis
  manifest     the training build's manifest   -> data/<version>/dataset/training
EOF
}

if [[ ${1-} == -h || ${1-} == --help ]]; then
    usage
    exit 0
fi

(($#)) || set -- coverage metadata selection stats labels swim verdicts manifest

for name in "$@"; do
    case "$name" in
        coverage | metadata | swim) download_one "$name" "data/analysis/$name" ;;
        selection | stats | labels) download_one "$version-$name" "data/$version/analysis/$name" ;;
        verdicts) download_one "$version-verdicts" "data/$version/analysis" shares ;;
        manifest)
            # Only the manifest comes down, never the build's crops
            (cd "$(dirname "$0")" && uv run --group digitalhub python -c \
                'import sys, digitalhub as dh; from dhub import store
store.download_manifest(dh.get_or_create_project(sys.argv[1]))' "$project")
            ;;
        *)
            echo "nothing is published under \`$name\`" >&2
            usage >&2
            exit 1
            ;;
    esac
done
