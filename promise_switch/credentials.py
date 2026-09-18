"""Centralized, secret-safe credential loading for Promise Switch."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ROOT_DOTENV_PATH = REPOSITORY_ROOT / ".env"


def load_repository_dotenv() -> None:
    """Load the deterministic repository-root dotenv without replacing shell values."""
    load_dotenv(dotenv_path=ROOT_DOTENV_PATH, override=False)


def api_key_from_environment(env_name: str) -> str:
    """Return a stripped credential after loading the root dotenv, or an empty string."""
    load_repository_dotenv()
    return os.environ.get(env_name, "").strip()


def require_api_key(env_name: str) -> str:
    """Resolve a required credential without including its value in any error."""
    value = api_key_from_environment(env_name)
    if not value:
        raise RuntimeError(f"{env_name} is not set")
    return value
