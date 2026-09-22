#!/usr/bin/env python
"""Publish the annotation's verdicts on DigitalHub, under the current version."""

from __future__ import annotations

import argparse

import digitalhub as dh
from dhub import store
from omegaconf import OmegaConf

from mars_multisensor_dataset.common.paths import CONFIGS_ROOT


def main() -> int:
    """Upload the verdicts the annotation notebook wrote.

    Returns:
        code: A process exit code, zero once they are up.
    """
    argparse.ArgumentParser(description=__doc__).parse_args()
    platform = OmegaConf.load(CONFIGS_ROOT / "digitalhub.yaml")
    store.upload_verdicts(dh.get_or_create_project(platform.project))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
