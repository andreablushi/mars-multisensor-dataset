"""Where the project lives on disk, and the dataset version both halves write under."""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]

CONFIGS_ROOT = REPO_ROOT / "configs"
DATA_ROOT = REPO_ROOT / "data"
VERSION_PATH = CONFIGS_ROOT / "version.yaml"
VERSION = OmegaConf.load(VERSION_PATH).version if VERSION_PATH.is_file() else None
VERSION_ROOT = DATA_ROOT / VERSION if VERSION else DATA_ROOT
