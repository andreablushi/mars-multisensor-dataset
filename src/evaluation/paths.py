"""Where the evaluation dataset is settled from, and where its labels are written."""

from __future__ import annotations

from common.paths import CONFIGS_ROOT, DATA_ROOT

EVALUATION_CONFIGS_ROOT = CONFIGS_ROOT / "evaluation"
ANALYSIS_CONFIG_PATH = EVALUATION_CONFIGS_ROOT / "analysis.yaml"
BUILDING_CONFIG_PATH = EVALUATION_CONFIGS_ROOT / "building.yaml"

EVALUATION_ROOT = DATA_ROOT / "evaluation"
FEATURES_PATH = EVALUATION_ROOT / "features.jsonl"
LABELS_ROOT = EVALUATION_ROOT / "labels"
LABELS_NAME = "labels.parquet"
