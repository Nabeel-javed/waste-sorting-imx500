from __future__ import annotations

from pathlib import Path

import yaml


UNKNOWN_BIN = "Unknown / no bin mapping"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BINS_CONFIG_PATH = PROJECT_ROOT / "configs" / "bins.yaml"


def _load_bin_mapping() -> dict[str, str]:
    if not BINS_CONFIG_PATH.exists():
        return {}
    with BINS_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return {str(key): str(value) for key, value in data.items()}


BIN_MAPPING = _load_bin_mapping()


def get_recommended_bin(class_name: str) -> str:
    """Return the mapped bin or UNKNOWN_BIN for classes outside this project."""
    return BIN_MAPPING.get(class_name, UNKNOWN_BIN)


def is_known_class(class_name: str) -> bool:
    return class_name in BIN_MAPPING


def format_recommendation(class_name: str, confidence: float) -> str:
    return f"{class_name} ({confidence:.2f}) -> {get_recommended_bin(class_name)}"

