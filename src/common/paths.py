"""Where the project lives on disk, which both halves write under."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_ROOT = REPO_ROOT / "configs"
DATA_ROOT = REPO_ROOT / "data"
