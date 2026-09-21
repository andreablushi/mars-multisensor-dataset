"""The one table every panel writes its rows into."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

import ipywidgets as widgets
import pandas as pd

from analysis.visualization.common.models.tables import Row


def written(title: str, headings: Sequence[str], rows: Sequence[Row]) -> widgets.HTML:
    """Write a table out as a panel."""
    frame = pd.DataFrame(rows, columns=list(headings))
    return widgets.HTML(f"<b>{escape(title)}</b>{frame.to_html(index=False, border=0)}")
