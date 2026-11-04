"""Settings come from environment variables (KP_*). Nothing here is required."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


@dataclass
class Settings:
    home: Path = field(default_factory=lambda: Path(_env("KP_HOME", ".kernelpilot")))
    db_path: Path | None = None
    # auto | triton | sim
    backend: str = field(default_factory=lambda: _env("KP_BACKEND", "auto"))
    # subprocess | inprocess. default depends on backend (see backends.make_backend)
    isolation: str = field(default_factory=lambda: _env("KP_ISOLATION", ""))
    timeout_s: float = field(default_factory=lambda: _env_float("KP_TIMEOUT", 120.0))
    seed: int = field(default_factory=lambda: _env_int("KP_SEED", 0))
    use_cache: bool = field(
        default_factory=lambda: _env("KP_CACHE", "1") not in ("0", "false", "no")
    )
    warmup: int = field(default_factory=lambda: _env_int("KP_WARMUP", 10))
    reps: int = field(default_factory=lambda: _env_int("KP_REPS", 50))
    # none | anthropic | openai
    llm_provider: str = field(default_factory=lambda: _env("KP_LLM_PROVIDER", "none"))
    llm_model: str = field(default_factory=lambda: _env("KP_LLM_MODEL", ""))
    log_level: str = field(default_factory=lambda: _env("KP_LOG_LEVEL", "WARNING"))
    log_json: bool = field(default_factory=lambda: _env("KP_LOG_JSON", "0") in ("1", "true"))

    def __post_init__(self) -> None:
        if self.db_path is None:
            self.db_path = Path(_env("KP_DB", str(self.home / "kernelpilot.db")))
        if self.backend not in ("auto", "triton", "sim"):
            raise ValueError(f"KP_BACKEND must be auto, triton or sim (got {self.backend!r})")
        if self.isolation not in ("", "subprocess", "inprocess"):
            raise ValueError(
                f"KP_ISOLATION must be subprocess or inprocess (got {self.isolation!r})"
            )

    def ensure_home(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        assert self.db_path is not None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


def load_settings(**overrides: object) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        if v is not None:
            setattr(s, k, v)
    s.__post_init__()
    return s
