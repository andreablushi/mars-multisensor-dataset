"""Downloading one instrument set's metadata of one feature, start to finish."""

from __future__ import annotations

from analysis.metadata.fetchers.products import fetch_products
from analysis.metadata.ode import ODEClient
from analysis.models.job import Job, Outcome
from utils.disk.files import write_jsonl


def download(job: Job, client: ODEClient, loc: str) -> Outcome:
    """Download one instrument set's metadata and write it out.

    Args:
        job: The feature and instrument set to download.
        client: The shared ODE client.
        loc: Which products a feature box returns.

    Returns:
        The outcome, carrying the error when the job failed.
    """
    try:
        records = fetch_products(client, job.feature, job.instrument_set, loc)
        write_jsonl(job.output_path, records)
        return Outcome(job=job)
    except Exception as exc:
        return Outcome(job=job, error=exc)
