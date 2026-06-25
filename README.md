# financial-news-analytics

An end-to-end financial news analytics and stock movement prediction pipeline that combines real-time and historical news ingestion, FinBERT sentiment analysis, PostgreSQL storage, feature engineering, stock price ingestion, and ML dataset generation — supporting both live inference and large historical backfills.

---

## Project Overview

This project builds a complete supervised machine-learning pipeline for predicting stock price movements from financial news sentiment.

**News ingestion** uses two complementary sources:

- **[Finnhub](https://finnhub.io)** provides recent, real-time financial news for daily inference.
- **[GDELT](https://www.gdeltproject.org/)** provides large historical news archives for multi-year backfills, enabling rich ML training datasets that the Finnhub free tier alone cannot supply.

Both sources are normalised into the same schema and feed into the same downstream pipeline:

1. **FinBERT** (`ProsusAI/finbert`) classifies each article as `positive`, `neutral`, or `negative`.
2. **PostgreSQL** stores all structured data — articles, sentiment results, and stock prices.
3. **Feature Engineering** aggregates per-ticker, per-day sentiment signals into a 22-column feature vector.
4. **Yahoo Finance** (`yfinance`) supplies historical OHLCV stock prices.
5. **ML Dataset Builder** joins features with forward-looking price labels to produce an XGBoost-ready supervised dataset.
6. **Phase 7** will train and evaluate an XGBoost classifier for BUY / HOLD / SELL prediction.

| Phase   | Status        | Description                                                              |
| ------- | ------------- | ------------------------------------------------------------------------ |
| Phase 1 | ✅ Complete   | News ingestion (Finnhub real-time + GDELT historical) → `data/raw/`      |
| Phase 2 | ✅ Complete   | FinBERT sentiment analysis → `data/processed/`                           |
| Phase 3 | ✅ Complete   | PostgreSQL storage → `news_articles` + `sentiment_results`               |
| Phase 4 | ✅ Complete   | Feature engineering → `data/features/`                                   |
| Phase 5 | ✅ Complete   | Stock price ingestion (Yahoo Finance / yfinance) → `stock_prices`        |
| Phase 6 | ✅ Complete   | ML dataset builder — feature + label generation → `data/ml/`             |
| Phase 7 | 🚧 Upcoming   | ML training — XGBoost stock movement prediction                          |

---

## Data Sources

| Source           | Purpose                             |
| ---------------- | ----------------------------------- |
| **Finnhub**      | Real-time financial news ingestion  |
| **GDELT**        | Historical financial news backfill  |
| **Yahoo Finance**| Historical OHLCV stock prices       |
| **FinBERT**      | Financial sentiment classification  |
| **PostgreSQL**   | Persistent structured storage       |

---

## Key Features

- Real-time financial news ingestion via Finnhub
- Historical news backfill via GDELT (multi-year archives)
- FinBERT sentiment analysis (`positive` / `neutral` / `negative`)
- PostgreSQL storage with safe upsert semantics
- Historical OHLCV stock price ingestion via Yahoo Finance
- Per-ticker daily feature engineering (22 sentiment + rolling features)
- Historical date-range feature generation for large backfills
- Supervised ML dataset generation with 13 label columns
- XGBoost-ready output with BUY / HOLD / SELL classification targets
- Structured logging throughout every phase
- Comprehensive unit test suite (650+ tests, no external dependencies required)

---

## Current Dataset Statistics

The following statistics reflect the current historical dataset generated from Jan 2025 to Jun 2026.

| Metric                   | Value     |
| ------------------------ | --------- |
| News articles            | 41,456    |
| Sentiment records        | 41,456    |
| Stock price records      | 1,825     |
| Tickers tracked          | 5         |
| Feature rows             | 579       |
| Feature columns          | 22        |
| ML dataset rows          | 488       |
| Label columns            | 13        |
| Prediction classes       | BUY / HOLD / SELL |

The ML dataset builder generated **488 labelled samples** from **579 engineered feature rows**, with the remainder dropped due to insufficient forward price data at the end of the date range.

---

## Project Structure

```
financial-news-analytics/
├── scripts/
│   ├── fetch_news.py          # Phase 1: fetch real-time news from Finnhub
│   ├── fetch_gdelt_news.py    # Phase 1: fetch historical news from GDELT
│   ├── run_sentiment.py       # Phase 2: run FinBERT sentiment analysis
│   ├── load_to_db.py          # Phase 3: load processed CSVs into PostgreSQL
│   ├── generate_features.py   # Phase 4: generate ML feature dataset
│   ├── fetch_prices.py        # Phase 5: ingest stock prices from Yahoo Finance
│   └── build_ml_dataset.py    # Phase 6: build supervised ML dataset with labels
│
├── src/
│   ├── ingestion/
│   │   ├── news_client.py     # Finnhub API client (real-time)
│   │   └── gdelt_client.py    # GDELT API client (historical backfill)
│   ├── processing/
│   │   └── sentiment_analyzer.py  # FinBERT sentiment pipeline
│   ├── storage/
│   │   ├── database.py        # Engine, session factory, DDL
│   │   ├── models.py          # SQLAlchemy ORM models (incl. StockPrice)
│   │   └── repository.py      # Upsert, bulk insert, queries
│   ├── features/
│   │   └── feature_engineer.py    # Phase 4: feature computation + range backfill
│   ├── prices/
│   │   ├── price_client.py    # Phase 5: Yahoo Finance / yfinance client
│   │   └── price_repository.py    # Phase 5: stock_prices DB access layer
│   ├── ml/
│   │   └── dataset_builder.py     # Phase 6: ML dataset + label generation
│   └── utils/
│       ├── config.py          # Pydantic settings (env-based)
│       ├── logger.py          # Structured logging
│       └── rate_limiter.py    # Token bucket rate limiter
│
├── tests/
│   ├── conftest.py            # Shared fixtures (settings mock)
│   └── unit/
│       ├── test_news_client.py
│       ├── test_gdelt_client.py       # GDELT client unit tests
│       ├── test_sentiment_analyzer.py
│       ├── test_run_sentiment.py      # Phase 2 script tests (incl. --input-file)
│       ├── test_repository.py         # Phase 3: SQLite in-memory tests
│       ├── test_feature_engineer.py   # Phase 4: 80 unit tests
│       ├── test_price_client.py       # Phase 5: 62 unit tests (yfinance mocked)
│       ├── test_price_repository.py   # Phase 5: 60 unit tests (SQLite in-memory)
│       └── test_dataset_builder.py    # Phase 6: 113 unit tests (SQLite in-memory)
│
├── data/
│   ├── raw/                   # Phase 1 output: Finnhub + GDELT CSVs (gitignored)
│   │   └── gdelt/             # GDELT backfill output (gitignored)
│   ├── processed/             # Phase 2 output (gitignored)
│   ├── features/              # Phase 4 output (gitignored)
│   └── ml/                    # Phase 6 output (gitignored)
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

### 2. Configure environment of the Project

```bash
cp .env.example .env
# Edit .env and set FINNHUB_API_KEY
```

### 3. Run Phase 1 — Fetch news

**Option A: Finnhub (real-time, last N days)**

```bash
python scripts/fetch_news.py
```

Options:

| Flag                  | Default     | Description            |
| --------------------- | ----------- | ---------------------- |
| `--tickers AAPL TSLA` | from `.env` | Tickers to fetch       |
| `--days 7`            | from `.env` | Lookback window (days) |
| `--log-level INFO`    | from `.env` | Verbosity              |
| `--dry-run`           | —           | Print config and exit  |

Output: `data/raw/<TICKER>_news_<YYYY-MM-DD>.csv` + `data/raw/summary_<YYYY-MM-DD>.csv`

**Option B: GDELT (historical backfill)**

```bash
# Backfill a full year for all tickers
python scripts/fetch_gdelt_news.py --start-date 2025-01-01 --end-date 2025-12-31

# Specific tickers and date range
python scripts/fetch_gdelt_news.py --tickers AAPL TSLA NVDA \
    --start-date 2025-01-01 --end-date 2025-06-30
```

Output: `data/raw/gdelt/<TICKER>_gdelt_<START>_<END>.csv`

### 4. Run Phase 2 — Sentiment analysis

**Finnhub files (date-based discovery):**

```bash
python scripts/run_sentiment.py
python scripts/run_sentiment.py --date 2026-06-15
```

**GDELT backfill files (direct file mode):**

```bash
python scripts/run_sentiment.py --input-file data/raw/gdelt/NVDA_gdelt_2025-01-01_2025-12-31.csv
```

The `--input-file` flag processes any single CSV directly — ticker and output tag are inferred from the filename. For GDELT files the output tag is the start year (e.g. `NVDA_sentiment_2025.csv`).

Options:

| Flag                       | Default            | Description                                         |
| -------------------------- | ------------------ | --------------------------------------------------- |
| `--tickers AAPL TSLA`      | all CSVs for today | Tickers to process (discovery mode)                 |
| `--date 2026-06-15`        | today              | Date tag of input CSVs (discovery mode)             |
| `--input-file PATH`        | —                  | Process a single CSV directly (skips discovery)     |
| `--model ProsusAI/finbert` | from `.env`        | Hugging Face model ID                               |
| `--batch-size 32`          | from `.env`        | Inference batch size                                |
| `--device auto`            | from `.env`        | `auto` / `cpu` / `cuda` / `mps`                     |
| `--log-level INFO`         | from `.env`        | Verbosity                                           |
| `--dry-run`                | —                  | Print config and exit                               |

Output: `data/processed/<TICKER>_sentiment_<tag>.csv` + `data/processed/sentiment_summary_<tag>.csv`

### 5. Run Phase 3 — Load to PostgreSQL

```bash
python scripts/load_to_db.py --create-tables
```

Options:

| Flag                            | Default            | Description                                     |
| ------------------------------- | ------------------ | ----------------------------------------------- |
| `--tickers AAPL TSLA`           | all CSVs for today | Tickers to load                                 |
| `--date 2026-06-16`             | today              | Date tag of processed CSVs                      |
| `--model-name ProsusAI/finbert` | from `.env`        | Model name stored in DB                         |
| `--create-tables`               | —                  | Run `CREATE TABLE IF NOT EXISTS` before loading |
| `--log-level INFO`              | from `.env`        | Verbosity                                       |
| `--dry-run`                     | —                  | Print config and exit                           |

Output: rows upserted into `news_articles` and `sentiment_results` tables.

### 6. Run Phase 4 — Feature engineering

```bash
# Single day
python scripts/generate_features.py --date 2026-06-16

# Historical date-range backfill
python scripts/generate_features.py --start-date 2025-01-01 --end-date 2025-12-31
```

Options:

| Flag                  | Default           | Description                                      |
| --------------------- | ----------------- | ------------------------------------------------ |
| `--tickers AAPL TSLA` | all tickers in DB | Tickers to generate features for                 |
| `--date 2026-06-16`   | today             | Target date (single-day mode)                    |
| `--start-date`        | —                 | Start of date range (range mode)                 |
| `--end-date`          | —                 | End of date range (range mode)                   |
| `--output-dir PATH`   | `data/features/`  | Directory for output CSV                         |
| `--lookback-days 7`   | `7`               | History window for rolling features              |
| `--log-level INFO`    | from `.env`       | Verbosity                                        |
| `--dry-run`           | —                 | Print config and exit                            |

Output: `data/features/feature_dataset_<YYYY-MM-DD>.csv` (single day) or `data/features/feature_dataset_<START>_<END>.csv` (range)

### 7. Run Phase 5 — Stock price ingestion

```bash
# Create the stock_prices table (safe on existing DBs — uses IF NOT EXISTS)
python scripts/fetch_prices.py --create-tables

# Fetch one year of history for all tickers defined in .env
python scripts/fetch_prices.py --lookback-days 365

# Specific tickers with a fixed date range
python scripts/fetch_prices.py --tickers AAPL TSLA NVDA \
    --start-date 2025-01-01 --end-date 2026-01-01

# Dry-run: fetch from Yahoo Finance but skip all database writes
python scripts/fetch_prices.py --tickers AAPL --lookback-days 30 --dry-run

# One-shot: create tables then populate
python scripts/fetch_prices.py --create-tables --lookback-days 365
```

Options:

| Flag                        | Default             | Description                                      |
| --------------------------- | ------------------- | ------------------------------------------------ |
| `--tickers AAPL TSLA`       | from `.env`         | Ticker symbols to fetch                          |
| `--start-date 2025-01-01`   | today − lookback    | Inclusive start date                             |
| `--end-date 2026-01-01`     | today               | End date (exclusive per yfinance convention)     |
| `--lookback-days 365`       | `365`               | Days of history when `--start-date` is omitted  |
| `--create-tables`           | —                   | Run `CREATE TABLE IF NOT EXISTS` before fetching |
| `--dry-run`                 | —                   | Fetch data but skip all DB writes                |
| `--log-level INFO`          | from `.env`         | Verbosity                                        |

Output: rows upserted into the `stock_prices` table.

---

## Historical News Backfill (GDELT)

The Finnhub free tier limits lookback to approximately one year and caps daily request volume. For meaningful ML training, the model needs multiple years of labelled examples per ticker — far more than Finnhub alone can supply.

**GDELT** (Global Database of Events, Language, and Tone) is a free, open, real-time database of global news events. Its news-article index covers hundreds of thousands of financial news sources and extends back many years, making it ideal for building large historical training corpora.

The GDELT integration (`src/ingestion/gdelt_client.py`) enables:

- **Multi-year historical backfills** — fetch news for any date range, from months to years
- **Scalable ingestion** — rate-limited, batched requests with automatic retry
- **Identical downstream processing** — GDELT output uses the exact same column schema as Finnhub, so the same FinBERT pipeline, database loader, and feature engineer all work unchanged
- **Larger ML datasets** — more training samples lead to better generalisation and more reliable evaluation
- **Reproducible experiments** — fixed date ranges produce fully deterministic datasets

Both Finnhub and GDELT files feed into the same Phase 2 → 3 → 4 → 5 → 6 pipeline without any modification to the downstream code.

---

## Data Schemas

### Phase 1 — Raw news (`data/raw/<TICKER>_news_<date>.csv` or `data/raw/gdelt/<TICKER>_gdelt_<START>_<END>.csv`)

| Column         | Type           | Description                       |
| -------------- | -------------- | --------------------------------- |
| `ticker`       | str            | Ticker symbol                     |
| `source_id`    | str            | Article ID (Finnhub) or hash (GDELT) |
| `source_name`  | str            | Publisher name                    |
| `author`       | null           | Not provided by either source     |
| `title`        | str            | Article headline                  |
| `description`  | str            | Article summary                   |
| `url`          | str            | Article URL                       |
| `published_at` | datetime (UTC) | Publication timestamp             |
| `content`      | null           | Not provided by either source     |
| `fetched_at`   | datetime (UTC) | Fetch timestamp                   |

### Phase 2 — Sentiment-enriched (`data/processed/<TICKER>_sentiment_<tag>.csv`)

All Phase 1 columns plus:

| Column                 | Type           | Values                              |
| ---------------------- | -------------- | ----------------------------------- |
| `sentiment_label`      | str            | `positive` / `neutral` / `negative` |
| `sentiment_score`      | int            | `+1` / `0` / `-1`                   |
| `sentiment_confidence` | float          | Softmax probability in [0, 1]       |
| `analysed_at`          | str (ISO-8601) | UTC timestamp of inference          |

---

## Sentiment Analysis Model

**Model:** [ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)

FinBERT is a BERT model fine-tuned on financial news corpora for sentiment classification. It outperforms general-purpose BERT variants on financial text.

| Label    | Score | Meaning               |
| -------- | ----- | --------------------- |
| positive | +1    | Bullish signal        |
| neutral  | 0     | No directional signal |
| negative | -1    | Bearish signal        |

**Input text:** `"<title>. <description>"` (or just `<title>` if description is absent).

**Note:** The model is downloaded from Hugging Face on first run (~440 MB) and cached locally in `~/.cache/huggingface/`. Set `HF_HOME` to change the cache location.

---

## Configuration

All settings are managed via environment variables (or `.env`):

| Variable             | Default                    | Description                                |
| -------------------- | -------------------------- | ------------------------------------------ |
| `FINNHUB_API_KEY`    | **required**               | Finnhub API key                            |
| `TICKERS`            | `AAPL,TSLA,NVDA,MSFT,AMZN` | Default ticker list                        |
| `NEWS_LOOKBACK_DAYS` | `7`                        | Days of news history to fetch              |
| `LOG_LEVEL`          | `INFO`                     | Logging verbosity                          |
| `FINBERT_MODEL`      | `ProsusAI/finbert`         | Hugging Face model ID                      |
| `FINBERT_BATCH_SIZE` | `32`                       | Inference batch size                       |
| `FINBERT_DEVICE`     | `auto`                     | Compute device (`auto`/`cpu`/`cuda`/`mps`) |
| `DATABASE_URL`       | `None`                     | PostgreSQL connection URL (Phase 3 only)   |

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

### Phase 5 — Stock prices (`stock_prices`)

```
stock_prices
├── id              BIGINT PK (auto-increment)
├── ticker          VARCHAR(10)
├── trading_date    DATE          ← UNIQUE with ticker (dedup key)
├── open_price      FLOAT (nullable)
├── high_price      FLOAT (nullable)
├── low_price       FLOAT (nullable)
├── close_price     FLOAT (nullable)
├── adjusted_close  FLOAT (nullable — falls back to close_price if absent)
├── volume          BIGINT (nullable)
└── created_at      TIMESTAMPTZ (server default)

Indexes:
  ix_stock_prices_ticker       on ticker
  ix_stock_prices_ticker_date  on (ticker, trading_date)
```

Source: Yahoo Finance via `yfinance` with `auto_adjust=False`.  
Re-running `fetch_prices.py` is always safe — `ON CONFLICT DO UPDATE` refreshes OHLCV values in place.

---

## Full Pipeline Diagram

```
Finnhub API ──────────────────\
(real-time, Phase 1)           \
                                ├──► Phase 2: run_sentiment.py (FinBERT)
GDELT ─────────────────────────/         │
(historical backfill, Phase 1)           │
                                         ▼
                              data/processed/<TICKER>_sentiment_<tag>.csv
                                         │
                                         │  Phase 3: load_to_db.py
                                         ▼
                              news_articles + sentiment_results  (PostgreSQL)
                                         │
                                         │  Phase 4: generate_features.py
                                         ▼
                              data/features/feature_dataset_<date>.csv
                                         │
Yahoo Finance ───────────────────────────┤  Phase 5: fetch_prices.py
(OHLCV prices)                           │         ▼
                                         │    stock_prices  (PostgreSQL)
                                         │         │
                                         │  Phase 6: build_ml_dataset.py
                                         ▼
                              data/ml/ml_dataset_<date>.csv
                                         │
                                         │  Phase 7 (upcoming): XGBoost training
                                         ▼
                                   BUY / HOLD / SELL  prediction
```

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

**Historical backfill:** `generate_features.py` also supports a `--start-date` / `--end-date` range mode that iterates over every date in the window, producing a single combined CSV covering the full period. This makes it practical to generate months or years of training features from a historical GDELT backfill in a single command.

#### Exception hierarchy

```
FeatureEngineeringError          (base)
├── DataLoadError                (database connectivity / query failure)
└── FeatureGenerationError       (empty input / no articles on target date)
```

### Feature Definitions

Each row in the output dataset represents one ticker on one date.

#### Sentiment Features (11)

| Feature                | Description                                                     |
| ---------------------- | --------------------------------------------------------------- |
| `article_count`        | Total articles on the target date                               |
| `positive_count`       | Articles with `sentiment_label == "positive"`                   |
| `neutral_count`        | Articles with `sentiment_label == "neutral"`                    |
| `negative_count`       | Articles with `sentiment_label == "negative"`                   |
| `positive_ratio`       | `positive_count / article_count`                                |
| `neutral_ratio`        | `neutral_count / article_count`                                 |
| `negative_ratio`       | `negative_count / article_count`                                |
| `mean_sentiment_score` | Mean of `sentiment_score` (−1 / 0 / +1) on target date          |
| `sentiment_score_std`  | Sample std-dev of `sentiment_score` (0 for single-article days) |
| `sentiment_score_min`  | Min `sentiment_score` on target date                            |
| `sentiment_score_max`  | Max `sentiment_score` on target date                            |

#### Source Features (4)

| Feature                  | Description                                                        |
| ------------------------ | ------------------------------------------------------------------ |
| `unique_source_count`    | Number of distinct `source_name` values                            |
| `yahoo_article_count`    | Articles whose `source_name` contains `"yahoo"` (case-insensitive) |
| `benzinga_article_count` | Articles whose `source_name` contains `"benzinga"`                 |
| `cnbc_article_count`     | Articles whose `source_name` contains `"cnbc"`                     |

#### Time Features (3)

All windows end on (and include) the target date.

| Feature             | Window                                         |
| ------------------- | ---------------------------------------------- |
| `articles_last_24h` | Articles published on exactly the target date  |
| `articles_last_3d`  | Articles in `[target_date − 2 d, target_date]` |
| `articles_last_7d`  | Articles in `[target_date − 6 d, target_date]` |

#### Rolling Features (4)

Rolling means are computed from **daily aggregates** — each calendar day contributes one data point (its daily mean sentiment) regardless of article volume. This gives days equal weight and produces a smoother, less volume-biased signal.

| Feature                     | Window | Description                                    |
| --------------------------- | ------ | ---------------------------------------------- |
| `rolling_3d_mean_sentiment` | 3 days | Mean of the last 3 daily mean sentiment scores |
| `rolling_7d_mean_sentiment` | 7 days | Mean of the last 7 daily mean sentiment scores |
| `rolling_3d_article_volume` | 3 days | Total article count in the last 3 days         |
| `rolling_7d_article_volume` | 7 days | Total article count in the last 7 days         |

### Example Output

```
ticker,date,article_count,positive_count,neutral_count,negative_count,positive_ratio,neutral_ratio,negative_ratio,mean_sentiment_score,sentiment_score_std,sentiment_score_min,sentiment_score_max,unique_source_count,yahoo_article_count,benzinga_article_count,cnbc_article_count,articles_last_24h,articles_last_3d,articles_last_7d,rolling_3d_mean_sentiment,rolling_7d_mean_sentiment,rolling_3d_article_volume,rolling_7d_article_volume
AAPL,2026-06-16,246,69,109,68,0.280488,0.443089,0.276422,0.004065,0.816517,-1,1,12,87,42,18,246,531,1203,0.012341,0.008922,531,1203
TSLA,2026-06-16,183,42,91,50,0.229508,0.497268,0.273224,-0.043716,0.803214,-1,1,9,61,38,14,183,402,954,-0.018234,-0.011203,402,954
```

---

## Phase 6 — ML Dataset Builder

Phase 6 assembles the **supervised training dataset** that Phase 7 (XGBoost) consumes.  It joins the Phase 4 feature vectors with future stock price movements drawn from `stock_prices` to produce binary and multi-class labels for every (ticker, date) pair.

The builder supports both single-date and historical range modes. Running against the full historical feature dataset (579 rows spanning Jan 2025 – Jun 2026) produced **488 labelled ML samples** — the remaining 91 rows were dropped because insufficient forward price data was available at the end of the date range.

### Pipeline Architecture

```
data/features/
└── feature_dataset_<date>.csv     ← Phase 4 output
         │
         ▼
 MLDatasetBuilder.load_features()
         │
         ▼
 MLDatasetBuilder.load_prices()    ← PostgreSQL: stock_prices
         │
         ▼
 MLDatasetBuilder.generate_labels()
   ├── _compute_future_closes()    ← N-trading-day lookahead (index-based)
   ├── _compute_returns()          ← (future - today) / today
   ├── _compute_binary_labels()    ← 1 if return > 0 else 0
   └── _compute_direction_label()  ← BUY / HOLD / SELL from 5d return
         │
         ▼
 MLDatasetBuilder.build_dataset()  ← enforce column order
         │
         ▼
data/ml/
└── ml_dataset_<date>.csv          ← Phase 6 output → Phase 7 input
```

### Running Phase 6

```bash
# Build the ML dataset for today (reads matching feature_dataset_<today>.csv)
python scripts/build_ml_dataset.py

# Specific date
python scripts/build_ml_dataset.py --date 2026-06-16

# Custom output directory
python scripts/build_ml_dataset.py --date 2026-06-16 --output-dir /tmp/ml

# Extend the lookahead price window (default: 14 calendar days)
python scripts/build_ml_dataset.py --date 2026-06-16 --lookahead-days 21

# Dry-run: print config and exit without touching any data
python scripts/build_ml_dataset.py --dry-run
```

### Label Generation Workflow

For each `(ticker, date)` row in the feature dataset:

1. **Locate today's close** — look up `close_price` for `date` in `stock_prices`.
2. **Find future closes** — using a trading-day index (not calendar days), find the closing price N trading days ahead for N ∈ {1, 3, 5, 7}.
3. **Compute returns** — apply the formula below.
4. **Assign binary labels** — one label per horizon.
5. **Assign direction label** — one multi-class label from the 5-day return.

Rows for which any required future close is unavailable (e.g. data not yet ingested) are **logged and skipped** — they do not appear in the output dataset.

### Future Return Formulas

```
return_Nd = (close_future_N - close_today) / close_today
```

| Column        | Lookahead |
| ------------- | --------- |
| `return_1d`   | 1 trading day  |
| `return_3d`   | 3 trading days |
| `return_5d`   | 5 trading days |
| `return_7d`   | 7 trading days |

> **Trading-day indexing** — the "Nth trading day ahead" skips weekends and market holidays automatically because only dates present in `stock_prices` are considered.

### BUY / HOLD / SELL Definitions

| Label  | Condition           |
| ------ | ------------------- |
| `BUY`  | `return_5d > 0.02`  |
| `SELL` | `return_5d < -0.02` |
| `HOLD` | otherwise           |

The 2 % threshold on the 5-day return was chosen to filter out noise while still capturing meaningful directional moves.

### ML Dataset Schema

The output CSV contains **all Phase 4 feature columns** (23 columns) followed by **13 label columns** (36 total).

#### Label Columns (13)

| Column             | Type    | Description                                        |
| ------------------ | ------- | -------------------------------------------------- |
| `future_close_1d`  | float   | Closing price 1 trading day after target date      |
| `future_close_3d`  | float   | Closing price 3 trading days after target date     |
| `future_close_5d`  | float   | Closing price 5 trading days after target date     |
| `future_close_7d`  | float   | Closing price 7 trading days after target date     |
| `return_1d`        | float   | `(future_close_1d − close_today) / close_today`    |
| `return_3d`        | float   | `(future_close_3d − close_today) / close_today`    |
| `return_5d`        | float   | `(future_close_5d − close_today) / close_today`    |
| `return_7d`        | float   | `(future_close_7d − close_today) / close_today`    |
| `label_up_1d`      | int 0/1 | 1 if `return_1d > 0`                               |
| `label_up_3d`      | int 0/1 | 1 if `return_3d > 0`                               |
| `label_up_5d`      | int 0/1 | 1 if `return_5d > 0`                               |
| `label_up_7d`      | int 0/1 | 1 if `return_7d > 0`                               |
| `label_direction`  | str     | `BUY` / `HOLD` / `SELL` (from `return_5d`)         |

### Example Output Row

```
ticker,date,...(23 feature cols)...,future_close_1d,future_close_3d,future_close_5d,future_close_7d,return_1d,return_3d,return_5d,return_7d,label_up_1d,label_up_3d,label_up_5d,label_up_7d,label_direction
AAPL,2026-06-16,...,213.50,215.80,218.20,217.60,0.00704,0.01767,0.02898,0.02618,1,1,1,1,BUY
TSLA,2026-06-16,...,248.30,244.10,241.50,246.80,-0.00682,-0.02282,-0.03243,-0.01082,0,0,0,0,SELL
```

### Missing Data Handling

| Scenario | Behaviour |
| -------- | --------- |
| Ticker not in `stock_prices` | Row skipped, warning logged |
| Feature date not in `stock_prices` | Row skipped, warning logged |
| Insufficient future trading days | Row skipped, warning logged |
| All required future closes missing | `LabelGenerationError` raised |
| `NULL` close price in database | Treated as missing, row skipped |

---

## Next Phase — Phase 7: XGBoost Training

With 488 labelled samples and a 22-feature dataset ready, Phase 7 will:

- **Train an XGBoost classifier** on the `label_direction` (BUY / HOLD / SELL) target
- **Analyse feature importance** to understand which sentiment signals are most predictive
- **Cross-validate** with time-series-aware splits to avoid look-ahead bias
- **Tune hyperparameters** via grid or Bayesian search
- **Evaluate** with precision, recall, F1, and confusion matrix per class
- **Expose a prediction API** that takes a feature row and returns a direction signal
