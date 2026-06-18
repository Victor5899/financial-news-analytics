"""
Application configuration loaded from environment variables / .env file.

All settings are validated at import time via Pydantic. If required fields
are missing the process exits immediately with a human-readable message
rather than failing later with a cryptic error.

Usage
-----
    from src.utils.config import settings

    key   = settings.finnhub_api_key
    ticks = settings.tickers
"""

from __future__ import annotations

import sys

from typing import Optional

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings.

    Values are resolved in this order (highest priority first):
    1. Real environment variables
    2. .env file in the project root
    3. Defaults defined below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Required ────────────────────────────────────────────────────────────
    finnhub_api_key: str = Field(
        ...,
        description="Finnhub API key (https://finnhub.io/register)",
    )

    # ── Ticker universe ─────────────────────────────────────────────────────
    tickers: list[str] = Field(
        default=["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"],
        description="Comma-separated ticker symbols to track",
    )

    # ── Fetch parameters ─────────────────────────────────────────────────────
    news_lookback_days: int = Field(
        default=7,
        ge=1,
        le=365,
        description=(
            "Days back from today to fetch articles. "
            "Finnhub company-news supports up to ~1 year on the free tier."
        ),
    )

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging verbosity: DEBUG | INFO | WARNING | ERROR",
    )

    # ── Phase 2: FinBERT sentiment analysis ──────────────────────────────────
    finbert_model: str = Field(
        default="ProsusAI/finbert",
        description="Hugging Face model identifier for FinBERT sentiment analysis",
    )

    finbert_batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Number of articles per inference batch (higher = faster, more VRAM)",
    )

    finbert_device: str = Field(
        default="auto",
        description=(
            "Compute device for model inference. "
            "'auto' selects CUDA → MPS → CPU. "
            "Explicit values: 'cpu', 'cuda', 'mps'."
        ),
    )

    # ── Phase 3: PostgreSQL storage ──────────────────────────────────────────
    database_url: Optional[str] = Field(
        default=None,
        description=(
            "SQLAlchemy-compatible PostgreSQL connection URL. "
            "Required for Phase 3 (scripts/load_to_db.py). "
            "Example: postgresql://user:password@localhost:5432/financial_news"
        ),
    )

    # ── Validators ───────────────────────────────────────────────────────────
    @field_validator("tickers", mode="before")
    @classmethod
    def _parse_tickers(cls, raw: str | list[str]) -> list[str]:
        """Accept either a comma-separated string or a Python list."""
        if isinstance(raw, str):
            return [t.strip().upper() for t in raw.split(",") if t.strip()]
        return [t.strip().upper() for t in raw]

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalise_log_level(cls, raw: str) -> str:
        return raw.upper()

    @field_validator("finnhub_api_key", mode="after")
    @classmethod
    def _reject_placeholder(cls, key: str) -> str:
        if key in {"", "your_finnhub_api_key_here"}:
            raise ValueError(
                "FINNHUB_API_KEY is not set. "
                "Copy .env.example to .env and add your key from https://finnhub.io/register"
            )
        return key


def _load_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        lines = ["", "Configuration error — fix your .env file:", ""]
        for err in exc.errors():
            field = " → ".join(str(loc) for loc in err["loc"])
            lines.append(f"  {field}: {err['msg']}")
        lines += ["", "  Hint: copy .env.example to .env and fill in the values.", ""]
        sys.exit("\n".join(lines))


settings: Settings = _load_settings()
