"""Which observations are a lit look at the tile rather than a clip of its edge."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from math import inf

from analysis.coverage.models.coverage import SetCoverage
from analysis.metadata.loaders.observations import read_incidences
from analysis.models.instrument import PIXEL_KM2
from analysis.selector.artifacts import refused_pdsids
from analysis.selector.models.criteria import Criteria
from analysis.selector.models.search_grid import SearchGrid
from analysis.selector.models.track import Offered
from analysis.utils import mask as packing
from analysis.utils.tile_group import group_of_tile_named
from building.configs import crism, sharad


def admitted_observations(
    coverage: Sequence[SetCoverage],
    grid: SearchGrid,
    criteria: Criteria,
) -> tuple[Offered, Offered]:
    """Keep every lit observation big enough for the tile, and turn the rest away.

    Args:
        coverage: The tile's instrument sets, in any order.
        grid: The grid the tile is searched over.
        criteria: The ground and the lighting each instrument is asked for.

    Returns:
        admitted: What the tile keeps, with each set and the cells it fills.
        refused: What it turned away, carrying the same.
    """
    admitted: Offered = []
    refused: Offered = []
    tile = coverage[0].summary.tile
    incidences = read_incidences(group_of_tile_named(tile))
    emptied = refused_pdsids().get(tile, frozenset())
    for owner, instrument in enumerate(coverage):
        iid = instrument.summary.iid
        limit = (
            inf
            if iid == sharad.LAYOUT.instrument
            else criteria.solar_zenith.get(iid, inf)
        )
        floor = floor_km2(criteria, iid)
        for observation in instrument.events:
            cells = [
                cell
                for cell in packing.filled_cells(observation.mask).tolist()
                if cell in grid.inside
            ]
            if not cells:
                continue
            landed_km2 = len(cells) * grid.cell_km2
            lit = incidences.get(observation.pdsid, inf) <= limit
            measured = observation.pdsid not in emptied
            verdict = admitted if lit and measured and landed_km2 >= floor else refused
            verdict.append((observation, owner, cells))
    return whole_observations(admitted, refused, coverage)


def whole_observations(
    admitted: Offered, refused: Offered, coverage: Sequence[SetCoverage]
) -> tuple[Offered, Offered]:
    """Turn away every CRISM look whose other detector the tile did not admit.

    Args:
        admitted: What the tile keeps so far, with each set and the cells it fills.
        refused: What it turned away so far, carrying the same.
        coverage: The tile's instrument sets, which the owners index.

    Returns:
        admitted: What it keeps, every CRISM observation with both its detectors.
        refused: What it turned away, the lone halves among them.
    """
    crism_looks = [
        coverage[owner].summary.iid == crism.ODE["iid"] for _, owner, _ in admitted
    ]
    detectors: dict[str, set[str]] = defaultdict(set)
    for (observation, _, _), is_crism in zip(admitted, crism_looks, strict=True):
        if is_crism:
            parts = crism.NAMING.parts(observation.pdsid) or {}
            detectors[crism.NAMING.observation_id(observation.pdsid)].add(
                parts.get("detector", "")
            )
    whole = {name for name, held in detectors.items() if held >= set(crism.Detector)}
    lone = [
        is_crism and crism.NAMING.observation_id(one[0].pdsid) not in whole
        for one, is_crism in zip(admitted, crism_looks, strict=True)
    ]
    return (
        [one for one, half in zip(admitted, lone, strict=True) if not half],
        [*refused, *(one for one, half in zip(admitted, lone, strict=True) if half)],
    )


def floor_km2(criteria: Criteria, iid: str) -> float:
    """Return the ground one instrument's look has to land on a tile to count."""
    return criteria.admits.get(iid, 0.0) * PIXEL_KM2[iid]
