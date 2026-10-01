"""One instrument set's metadata over one group, downloaded start to finish."""

from __future__ import annotations

import httpx

from analysis.metadata.fetchers.products import fetch_products
from analysis.models.job import DownloadJob, Outcome
from common.disk.files import write_jsonl


def download_outcome(job: DownloadJob, client: httpx.Client, loc: str) -> Outcome:
    """Download one instrument set's metadata and write it out.

    Args:
        job: The group and instrument set to download.
        client: The shared ODE client.
        loc: Which products a group box returns.

    Returns:
        outcome: The outcome, carrying the error when the job failed.
    """
    try:
        records = fetch_products(client, job.group, job.instrument_set, loc)
        write_jsonl(job.output_path, records)
        return Outcome(job=job)
    except Exception as exc:
        return Outcome(job=job, error=exc)
