"""What the looks a feature keeps left on it, instrument by instrument."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

from analysis.selector.models.track import Track
from analysis.stats.models.feature import FeatureLooks, FeatureStats, InstrumentReach


def measured_feature(looks: FeatureLooks) -> FeatureStats:
    """Read what the instruments left on one feature, given the looks it keeps.

    Args:
        looks: Its timeline, the window it earned, and where its looks sit on it.

    Returns:
        stats: What it holds.
    """
    track, window, taken = looks.track, looks.window, looks.taken
    # What each instrument left inside the window, and which of them each cell holds
    cells_by_iid: dict[str, set[int]] = {}
    observations_by_iid: dict[str, int] = {}
    iids_by_cell: dict[int, set[str]] = {}
    for index in taken:
        iid = track.iids[track.owners[index]]
        cells_by_iid.setdefault(iid, set()).update(track.cells[index])
        observations_by_iid[iid] = observations_by_iid.get(iid, 0) + 1
        for cell in track.cells[index].tolist():
            iids_by_cell.setdefault(cell, set()).add(iid)
    overlaps: dict[tuple[str, ...], float] = {}
    for cell in sorted(iids_by_cell):
        instrument_names = tuple(sorted(iids_by_cell[cell]))
        overlaps[instrument_names] = (
            overlaps.get(instrument_names, 0.0) + track.grid.cell_km2
        )
    # A pixel is one size whether or not its look was chosen, so all are read
    pixel_km2: dict[str, float] = {}
    for index, owner in enumerate(track.owners):
        iid = track.iids[owner]
        observation = track.observations[index]
        if iid not in pixel_km2 and observation.pixels and observation.own_km2:
            pixel_km2[iid] = observation.own_km2 / observation.pixels
    return FeatureStats(
        window=window,
        iids=list(dict.fromkeys(track.iids)),
        offered=dict(Counter(track.iids[owner] for owner in track.owners)),
        pixel_km2=pixel_km2,
        reached={
            iid: InstrumentReach(
                km2=len(cells_reached) * track.grid.cell_km2,
                pixels=_pixels_landed(track, taken, iid),
                observations_taken=observations_by_iid[iid],
            )
            for iid, cells_reached in cells_by_iid.items()
        },
        overlaps=dict(sorted(overlaps.items(), key=lambda ground: -ground[1])),
    )


def ground_by_instrument_count(
    overlaps: Mapping[tuple[str, ...], float],
) -> dict[int, float]:
    """Add up the ground each number of instruments reaches at once.

    Args:
        overlaps: The ground each set of instruments reaches, counting a cell once.

    Returns:
        ground: The ground in square kilometres, by how many instruments reach it,
            fewest first.
    """
    summed: dict[int, float] = {}
    for instrument_names, km2 in overlaps.items():
        summed[len(instrument_names)] = summed.get(len(instrument_names), 0.0) + km2
    return dict(sorted(summed.items()))


def _pixels_landed(track: Track, taken: Sequence[int], iid: str) -> float | None:
    """Add up the pixels one instrument landed on the feature inside its window.

    Args:
        track: The feature's admissible observations on one time axis.
        taken: Where the observations it keeps sit on that axis.
        iid: The instrument to count.

    Returns:
        pixels: The pixels it landed there, or None when any of its observations carries
            none.
    """
    total = 0.0
    for index in taken:
        if track.iids[track.owners[index]] != iid:
            continue
        observation = track.observations[index]
        if observation.pixels is None or not observation.own_km2:
            return None
        ground_km2 = len(track.cells[index]) * track.grid.cell_km2
        total += observation.pixels * ground_km2 / observation.own_km2
    return total
