"""
Parameter Sync — Single source of truth for strategy parameters.
The Streamlit app writes here, Python scripts read from here.
"""

import json
import os

PARAMS_DIR = os.path.join(os.path.dirname(__file__), "saved_params")
os.makedirs(PARAMS_DIR, exist_ok=True)


def _path(strategy_key: str) -> str:
    return os.path.join(PARAMS_DIR, f"{strategy_key}.json")


def save_params(strategy_key: str, params: dict) -> str:
    """Save params to disk. Returns the file path."""
    path = _path(strategy_key)
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
    return path


def load_params(strategy_key: str) -> dict:
    """Load saved params. Returns empty dict if no saved file."""
    path = _path(strategy_key)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def has_saved_params(strategy_key: str) -> bool:
    return os.path.exists(_path(strategy_key))


def list_saved_params() -> list:
    """Returns list of (strategy_key, filepath) tuples."""
    results = []
    if os.path.exists(PARAMS_DIR):
        for f in os.listdir(PARAMS_DIR):
            if f.endswith(".json"):
                key = f[:-5]
                results.append((key, os.path.join(PARAMS_DIR, f)))
    return sorted(results)


def delete_saved_params(strategy_key: str):
    path = _path(strategy_key)
    if os.path.exists(path):
        os.remove(path)
