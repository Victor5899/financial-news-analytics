# financial-news-analytics

Real-time financial news sentiment analytics and stock movement prediction pipeline.

---

## Project Overview

This project ingests financial news articles via [Finnhub](https://finnhub.io), enriches them with FinBERT sentiment scores, and lays the groundwork for downstream stock movement prediction.

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | ✅ Complete | Finnhub news ingestion → `data/raw/` |
| Phase 2 | ✅ Complete | FinBERT sentiment analysis → `data/processed/` |
| Phase 3 | ✅ Complete | PostgreSQL storage → `news_articles` + `sentiment_results` |
| Phase 4 | ✅ Complete | Feature engineering → `data/features/` |
| Phase 5 | 🔜 Planned | XGBoost stock movement prediction |

---

## Project Structure

```
financial-news-analytics/
├── scripts/
│   ├── fetch_news.py          # Phase 1: fetch news from Finnhub
│   ├── run_sentiment.py       # Phase 2: run FinBERT sentiment analysis
│   ├── load_to_db.py          # Phase 3: load processed CSVs into PostgreSQL
│   └── generate_features.py   # Phase 4: generate ML feature dataset
│
├── src/
│   ├── ingestion/
│   │   └── news_client.py     # Finnhub API client
│   ├── processing/
│   │   └── sentiment_analyzer.py  # FinBERT sentiment pipeline
│   ├── storage/
│   │   ├── database.py        # Engine, session factory, DDL
│   │   ├── models.py          # SQLAlchemy ORM models
│   │   └── repository.py      # Upsert, bulk insert, queries
│   ├── features/
│   │   └── feature_engineer.py    # Phase 4: feature computation
│   └── utils/
│       ├── config.py          # Pydantic settings (env-based)
│       ├── logger.py          # Structured logging
│       └── rate_limiter.py    # Token bucket rate limiter
│
├── tests/
│   ├── conftest.py            # Shared fixtures (settings mock)
│   └── unit/
│       ├── test_news_client.py
│       ├── test_sentiment_analyzer.py
│       ├── test_repository.py         # Phase 3: SQLite in-memory tests
│       └── test_feature_engineer.py   # Phase 4: 80 unit tests
│
├── data/
│   ├── raw/                   # Phase 1 output (gitignored)
│   ├── processed/             # Phase 2 output (gitignored)
│   └── features/              # Phase 4 output (gitignored)
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

### 5. Run Phase 3 — Load to PostgreSQL

```bash
python scripts/load_to_db.py --create-tables
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--tickers AAPL TSLA` | all CSVs for today | Tickers to load |
| `--date 2026-06-16` | today | Date tag of processed CSVs |
| `--model-name ProsusAI/finbert` | from `.env` | Model name stored in DB |
| `--create-tables` | — | Run `CREATE TABLE IF NOT EXISTS` before loading |
| `--log-level INFO` | from `.env` | Verbosity |
| `--dry-run` | — | Print config and exit |

Output: rows upserted into `news_articles` and `sentiment_results` tables.

### 6. Run Phase 4 — Feature engineering

```bash
python scripts/generate_features.py --date 2026-06-16
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--tickers AAPL TSLA` | all tickers in DB | Tickers to generate features for |
| `--date 2026-06-16` | today | Target date |
| `--output-dir PATH` | `data/features/` | Directory for output CSV |
| `--lookback-days 7` | `7` | History window for rolling features |
| `--log-level INFO` | from `.env` | Verbosity |
| `--dry-run` | — | Print config and exit |

Output: `data/features/feature_dataset_<YYYY-MM-DD>.csv`

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
| `DATABASE_URL` | `None` | PostgreSQL connection URL (Phase 3 only) |

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest                          # all tests
pytest -v                       # verbose
pytest tests/unit/test_sentiment_analyzer.py   # Phase 2 tests only
pytest --cov=src --cov-report=term-missing     # with coverage
```

All unit tests mock the Hugging Face pipeline and use an in-memory SQLite database — no model download and no PostgreSQL instance are needed for testing.

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

---

## Database Setup (Phase 3)

```bash
# Start a local PostgreSQL instance (example with Docker)
docker run -d \
  --name finews-pg \
  -e POSTGRES_DB=financial_news \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  postgres:16

# Add to .env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/financial_news

# Create tables + load first batch
python scripts/load_to_db.py --create-tables
```

### Database Schema

```
news_articles
├── id              BIGINT PK (auto-increment)
├── ticker          VARCHAR(10)
├── source_id       VARCHAR(64)
├── source_name     VARCHAR(255)
├── author          TEXT (nullable)
├── title           TEXT
├── description     TEXT (nullable)
├── url             TEXT  ← UNIQUE (dedup key)
├── published_at    TIMESTAMPTZ
├── content         TEXT (nullable)
├── fetched_at      TIMESTAMPTZ
└── created_at      TIMESTAMPTZ (server default)

sentiment_results
├── id                   BIGINT PK (auto-increment)
├── article_id           BIGINT FK → news_articles.id (CASCADE)
├── model_name           VARCHAR(255)        ← UNIQUE with article_id
├── sentiment_label      VARCHAR(10)
├── sentiment_score      SMALLINT (-1, 0, +1)
├── sentiment_confidence FLOAT
├── analysed_at          TIMESTAMPTZ
└── created_at           TIMESTAMPTZ (server default)
```

Re-running `load_to_db.py` is always safe — both tables use `ON CONFLICT DO UPDATE` (PostgreSQL) or SELECT-then-UPDATE (SQLite/tests), so existing rows are refreshed rather than duplicated.

---

## Phase 4 — Feature Engineering

### Architecture

`FeatureEngineer` is the central class in `src/features/feature_engineer.py`. It follows the same patterns as the rest of the pipeline: repository pattern, typed exceptions, structured logging, and clean separation between I/O and computation.

```
PostgreSQL
  news_articles + sentiment_results
          │
          │  load_data()  (SQL JOIN → pandas DataFrame)
          ▼
  raw_df  (ticker, source_name, published_at, date,
           sentiment_label, sentiment_score, sentiment_confidence)
          │
          │  generate_features()  (per-ticker feature computation)
          ▼
  features_df  (one row per ticker — 25 columns)
          │
          │  save_features()
          ▼
  data/features/feature_dataset_<date>.csv
```

#### Exception hierarchy

```
FeatureEngineeringError          (base)
├── DataLoadError                (database connectivity / query failure)
└── FeatureGenerationError       (empty input / no articles on target date)
```

### Feature Definitions

Each row in the output dataset represents one ticker on one date.

#### Sentiment Features (11)

| Feature | Description |
|---------|-------------|
| `article_count` | Total articles on the target date |
| `positive_count` | Articles with `sentiment_label == "positive"` |
| `neutral_count` | Articles with `sentiment_label == "neutral"` |
| `negative_count` | Articles with `sentiment_label == "negative"` |
| `positive_ratio` | `positive_count / article_count` |
| `neutral_ratio` | `neutral_count / article_count` |
| `negative_ratio` | `negative_count / article_count` |
| `mean_sentiment_score` | Mean of `sentiment_score` (−1 / 0 / +1) on target date |
| `sentiment_score_std` | Sample std-dev of `sentiment_score` (0 for single-article days) |
| `sentiment_score_min` | Min `sentiment_score` on target date |
| `sentiment_score_max` | Max `sentiment_score` on target date |

#### Source Features (4)

| Feature | Description |
|---------|-------------|
| `unique_source_count` | Number of distinct `source_name` values |
| `yahoo_article_count` | Articles whose `source_name` contains `"yahoo"` (case-insensitive) |
| `benzinga_article_count` | Articles whose `source_name` contains `"benzinga"` |
| `cnbc_article_count` | Articles whose `source_name` contains `"cnbc"` |

#### Time Features (3)

All windows end on (and include) the target date.

| Feature | Window |
|---------|--------|
| `articles_last_24h` | Articles published on exactly the target date |
| `articles_last_3d` | Articles in `[target_date − 2 d, target_date]` |
| `articles_last_7d` | Articles in `[target_date − 6 d, target_date]` |

#### Rolling Features (4)

Rolling means are computed from **daily aggregates** — each calendar day contributes one data point (its daily mean sentiment) regardless of article volume. This gives days equal weight and produces a smoother, less volume-biased signal.

| Feature | Window | Description |
|---------|--------|-------------|
| `rolling_3d_mean_sentiment` | 3 days | Mean of the last 3 daily mean sentiment scores |
| `rolling_7d_mean_sentiment` | 7 days | Mean of the last 7 daily mean sentiment scores |
| `rolling_3d_article_volume` | 3 days | Total article count in the last 3 days |
| `rolling_7d_article_volume` | 7 days | Total article count in the last 7 days |

### Example Output

```
ticker,date,article_count,positive_count,neutral_count,negative_count,positive_ratio,neutral_ratio,negative_ratio,mean_sentiment_score,sentiment_score_std,sentiment_score_min,sentiment_score_max,unique_source_count,yahoo_article_count,benzinga_article_count,cnbc_article_count,articles_last_24h,articles_last_3d,articles_last_7d,rolling_3d_mean_sentiment,rolling_7d_mean_sentiment,rolling_3d_article_volume,rolling_7d_article_volume
AAPL,2026-06-16,246,69,109,68,0.280488,0.443089,0.276422,0.004065,0.816517,-1,1,12,87,42,18,246,531,1203,0.012341,0.008922,531,1203
TSLA,2026-06-16,183,42,91,50,0.229508,0.497268,0.273224,-0.043716,0.803214,-1,1,9,61,38,14,183,402,954,-0.018234,-0.011203,402,954
```
