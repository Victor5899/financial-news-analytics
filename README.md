# financial-news-analytics

Real-time financial news sentiment analytics and stock movement prediction pipeline.

---

## Project Overview

This project ingests financial news articles via [Finnhub](https://finnhub.io), enriches them with FinBERT sentiment scores, and lays the groundwork for downstream stock movement prediction.

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Finnhub news ingestion → `data/raw/` |
| Phase 2 | ✅ Complete | FinBERT sentiment analysis → `data/processed/` |
| Phase 3 | 🔜 Planned | Feature engineering + ML model |

---

## Project Structure

```
financial-news-analytics/
├── scripts/
│   ├── fetch_news.py          # Phase 1: fetch news from Finnhub
│   └── run_sentiment.py       # Phase 2: run FinBERT sentiment analysis
│
├── src/
│   ├── ingestion/
│   │   └── news_client.py     # Finnhub API client
│   ├── processing/
│   │   └── sentiment_analyzer.py  # FinBERT sentiment pipeline
│   └── utils/
│       ├── config.py          # Pydantic settings (env-based)
│       ├── logger.py          # Structured logging
│       └── rate_limiter.py    # Token bucket rate limiter
│
├── tests/
│   ├── conftest.py            # Shared fixtures (settings mock)
│   └── unit/
│       ├── test_news_client.py
│       └── test_sentiment_analyzer.py
│
├── data/
│   ├── raw/                   # Phase 1 output (gitignored)
│   └── processed/             # Phase 2 output (gitignored)
│
├── .env.example               # Environment variable template
├── requirements.txt           # Runtime dependencies
├── requirements-dev.txt       # Dev/test dependencies
└── pyproject.toml             # Project metadata + tool config
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd financial-news-analytics
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set FINNHUB_API_KEY
```

### 3. Run Phase 1 — Fetch news

```bash
python scripts/fetch_news.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--tickers AAPL TSLA` | from `.env` | Tickers to fetch |
| `--days 7` | from `.env` | Lookback window (days) |
| `--log-level INFO` | from `.env` | Verbosity |
| `--dry-run` | — | Print config and exit |

Output: `data/raw/<TICKER>_news_<YYYY-MM-DD>.csv` + `data/raw/summary_<YYYY-MM-DD>.csv`

### 4. Run Phase 2 — Sentiment analysis

```bash
python scripts/run_sentiment.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--tickers AAPL TSLA` | all CSVs for today | Tickers to process |
| `--date 2026-06-15` | today | Date tag of input CSVs |
| `--model ProsusAI/finbert` | from `.env` | Hugging Face model ID |
| `--batch-size 32` | from `.env` | Inference batch size |
| `--device auto` | from `.env` | `auto` / `cpu` / `cuda` / `mps` |
| `--log-level INFO` | from `.env` | Verbosity |
| `--dry-run` | — | Print config and exit |

Output: `data/processed/<TICKER>_sentiment_<YYYY-MM-DD>.csv` + `data/processed/sentiment_summary_<YYYY-MM-DD>.csv`

---

## Data Schemas

### Phase 1 — Raw news (`data/raw/<TICKER>_news_<date>.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | str | Ticker symbol |
| `source_id` | str | Finnhub article ID |
| `source_name` | str | Publisher name |
| `author` | null | Not provided by Finnhub |
| `title` | str | Article headline |
| `description` | str | Article summary |
| `url` | str | Article URL |
| `published_at` | datetime (UTC) | Publication timestamp |
| `content` | null | Not provided by Finnhub |
| `fetched_at` | datetime (UTC) | Fetch timestamp |

### Phase 2 — Sentiment-enriched (`data/processed/<TICKER>_sentiment_<date>.csv`)

All Phase 1 columns plus:

| Column | Type | Values |
|--------|------|--------|
| `sentiment_label` | str | `positive` / `neutral` / `negative` |
| `sentiment_score` | int | `+1` / `0` / `-1` |
| `sentiment_confidence` | float | Softmax probability in [0, 1] |
| `analysed_at` | str (ISO-8601) | UTC timestamp of inference |

---

## Sentiment Analysis Model

**Model:** [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)

FinBERT is a BERT model fine-tuned on financial news corpora for sentiment classification. It outperforms general-purpose BERT variants on financial text.

| Label | Score | Meaning |
|-------|-------|---------|
| positive | +1 | Bullish signal |
| neutral | 0 | No directional signal |
| negative | -1 | Bearish signal |

**Input text:** `"<title>. <description>"` (or just `<title>` if description is absent).

**Note:** The model is downloaded from Hugging Face on first run (~440 MB) and cached locally in `~/.cache/huggingface/`. Set `HF_HOME` to change the cache location.

---

## Configuration

All settings are managed via environment variables (or `.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `FINNHUB_API_KEY` | **required** | Finnhub API key |
| `TICKERS` | `AAPL,TSLA,NVDA,MSFT,AMZN` | Default ticker list |
| `NEWS_LOOKBACK_DAYS` | `7` | Days of news history to fetch |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `FINBERT_MODEL` | `ProsusAI/finbert` | Hugging Face model ID |
| `FINBERT_BATCH_SIZE` | `32` | Inference batch size |
| `FINBERT_DEVICE` | `auto` | Compute device (`auto`/`cpu`/`cuda`/`mps`) |

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest                          # all tests
pytest -v                       # verbose
pytest tests/unit/test_sentiment_analyzer.py   # Phase 2 tests only
pytest --cov=src --cov-report=term-missing     # with coverage
```

All unit tests mock the Hugging Face pipeline — no model download is needed for testing.

---

## Development

```bash
# Lint
ruff check src/ tests/ scripts/

# Type check
mypy src/

# Format
ruff format src/ tests/ scripts/
```

---

## GPU / Apple Silicon

Phase 2 automatically detects and uses available hardware:

- **CUDA GPU** — install `torch` with CUDA support: `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- **Apple Silicon MPS** — standard `torch` installation works out of the box
- **CPU** — default fallback; slower but always available

Set `FINBERT_DEVICE=cpu` to force CPU inference regardless of available hardware.
