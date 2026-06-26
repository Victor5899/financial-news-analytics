# financial-news-analytics

An end-to-end financial news analytics and stock movement prediction pipeline that combines real-time and historical news ingestion, FinBERT sentiment analysis, PostgreSQL storage, feature engineering, stock price ingestion, and ML dataset generation — supporting both live inference and large historical backfills.

---

## Project Overview

This project builds a complete supervised machine-learning pipeline for predicting short-term stock price movements from financial news and technical analysis.

The pipeline has seven phases:

1. **News ingestion** — historical financial news is collected at scale using [GDELT](https://www.gdeltproject.org/) (multi-year archives) and [Finnhub](https://finnhub.io) (real-time feed). Both sources are normalised into the same schema and feed the same downstream pipeline.
2. **Sentiment analysis** — [FinBERT](https://huggingface.co/ProsusAI/finbert) (`ProsusAI/finbert`) classifies each article as `positive`, `neutral`, or `negative`.
3. **Storage** — [PostgreSQL](https://www.postgresql.org/) stores articles, sentiment results, and OHLCV stock prices with safe upsert semantics.
4. **Feature engineering** — per-ticker, per-day feature vectors are computed from sentiment aggregates and 19 technical indicators derived from OHLCV prices (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, rolling volatility, price returns, and volume features). The result is a 41-feature ML-ready dataset.
5. **Stock price ingestion** — [Yahoo Finance](https://finance.yahoo.com/) (`yfinance`) supplies historical OHLCV data that feeds both the technical indicator computation (Phase 4) and the forward-looking label generation (Phase 6).
6. **ML dataset builder** — feature vectors are joined with future price movements to produce binary and multi-class labels across 1-, 3-, 5-, and 7-trading-day horizons.
7. **Model training** — an [XGBoost](https://xgboost.readthedocs.io/) multi-class classifier is trained on the labelled dataset to predict BUY / HOLD / SELL signals, with a full evaluation suite and serialised model artifacts.

| Phase   | Status      | Description                                                              |
| ------- | ----------- | ------------------------------------------------------------------------ |
| Phase 1 | ✅ Complete | Historical Financial News Ingestion (Finnhub + GDELT) → `data/raw/`     |
| Phase 2 | ✅ Complete | FinBERT Sentiment Analysis → `data/processed/`                          |
| Phase 3 | ✅ Complete | PostgreSQL Data Storage → `news_articles` + `sentiment_results`         |
| Phase 4 | ✅ Complete | Feature Engineering (41 features) → `data/features/`                   |
| Phase 5 | ✅ Complete | Stock Price Ingestion (Yahoo Finance) → `stock_prices`                  |
| Phase 6 | ✅ Complete | ML Dataset Builder — labelled supervised dataset → `data/ml/`           |
| Phase 7 | ✅ Complete | XGBoost Model Training & Prediction → `artifacts/`                     |

---

## Technologies

| Technology       | Role                                             |
| ---------------- | ------------------------------------------------ |
| **Python**       | Core implementation language                     |
| **PostgreSQL**   | Persistent structured storage for all data       |
| **GDELT**        | Historical financial news ingestion (multi-year) |
| **Finnhub**      | Real-time financial news ingestion               |
| **Yahoo Finance**| Historical OHLCV stock price data (`yfinance`)  |
| **FinBERT**      | Financial sentiment classification               |
| **Transformers** | Hugging Face pipeline for FinBERT inference      |
| **PyTorch**      | FinBERT backend; GPU/MPS acceleration support    |
| **Pandas**       | Data manipulation and feature computation        |
| **NumPy**        | Numerical operations                             |
| **Scikit-learn** | Train/test split, label encoding, metrics        |
| **XGBoost**      | Gradient-boosted tree classifier                 |
| **Matplotlib**   | Feature importance bar-chart visualisation       |
| **Joblib**       | Model artifact serialisation                     |

---

## Key Features

- Multi-year historical news backfill via GDELT; real-time ingestion via Finnhub
- FinBERT sentiment analysis (`positive` / `neutral` / `negative`)
- PostgreSQL storage with safe upsert semantics across all three tables
- 41 engineered ML features per ticker per day:
  - Sentiment aggregates, source diversity, and time-window article counts
  - Rolling sentiment means (3-day and 7-day)
  - Trend indicators: SMA 10/20, EMA 10/20
  - Momentum indicators: RSI 14, MACD line, MACD signal, MACD histogram
  - Volatility indicators: Bollinger Bands (upper/lower/width), ATR 14, 20-day rolling volatility
  - Price return features: 1-day, 5-day, 10-day percentage changes
  - Volume features: volume change, 5-day average volume, volume ratio
- Technical indicators computed with **pandas only** — no external TA libraries
- Historical date-range feature backfill in a single command
- Supervised ML dataset with 13 label columns across four horizons (1d, 3d, 5d, 7d)
- XGBoost multi-class classifier: BUY / HOLD / SELL
- Stratified 80/20 train/test split with full evaluation suite
- Feature importance ranking: top-20 bar-chart PNG and ranked CSV
- Self-contained model artifact (model + encoder + feature list) via joblib
- Prediction API: CSV file, DataFrame, or single feature vector input
- Structured logging throughout every phase
- 870+ unit tests — no model download and no PostgreSQL instance required

---

## Current Dataset Statistics

The following statistics reflect the historical dataset generated from Jan 2025 to Jun 2026.

| Metric                   | Value             |
| ------------------------ | ----------------- |
| News articles            | 41,456            |
| Sentiment records        | 41,456            |
| Stock price records      | 1,825             |
| Tickers tracked          | 5                 |
| Feature rows             | 579               |
| Feature columns          | 41                |
| ML dataset rows          | 488               |
| Usable training samples  | 471               |
| Label columns            | 13                |
| Prediction classes       | BUY / HOLD / SELL |

The ML dataset builder generated **488 labelled samples** from **579 engineered feature rows**; the remainder were dropped due to insufficient forward price data at the end of the date range.

---

## Model Performance

The XGBoost classifier was trained on 471 labelled samples across 5 technology stocks using 41 engineered features.

| Metric           | Score   |
| ---------------- | ------- |
| Accuracy         | 51.58%  |
| Macro Precision  | 52.17%  |
| Macro Recall     | 51.45%  |
| Macro F1         | 51.41%  |

**Classes:** BUY · HOLD · SELL

**Note:** Predicting daily stock direction is an inherently noisy task. A 51.58% accuracy represents a meaningful signal above the naive baseline for a three-class problem (~33%), particularly given the short training window and the absence of hyperparameter tuning.

---

## Model Improvements

The project evolved from a **sentiment-only baseline** into a **hybrid sentiment + technical analysis model**:

| Model Version         | Features                                | Accuracy  |
| --------------------- | --------------------------------------- | --------- |
| Sentiment-only        | 22 sentiment and rolling features       | ~29%      |
| Sentiment + Technical | 41 features (sentiment + 19 indicators) | **51.58%** |

Adding technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, price returns, and volume features) increased model accuracy by approximately **+22 percentage points**. This confirms that price-derived signals carry strong predictive information that complements news sentiment alone.

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
│   ├── build_ml_dataset.py    # Phase 6: build supervised ML dataset with labels
│   ├── train_model.py         # Phase 7: train XGBoost classifier
│   └── predict.py             # Phase 7: run inference with saved model
│
├── src/
│   ├── ingestion/
│   │   ├── news_client.py         # Finnhub API client (real-time)
│   │   └── gdelt_client.py        # GDELT API client (historical backfill)
│   ├── processing/
│   │   └── sentiment_analyzer.py  # FinBERT sentiment pipeline
│   ├── storage/
│   │   ├── database.py            # Engine, session factory, DDL
│   │   ├── models.py              # SQLAlchemy ORM models (incl. StockPrice)
│   │   └── repository.py          # Upsert, bulk insert, queries
│   ├── features/
│   │   └── feature_engineer.py    # Phase 4: sentiment + technical indicator features
│   ├── prices/
│   │   ├── price_client.py        # Phase 5: Yahoo Finance / yfinance client
│   │   └── price_repository.py    # Phase 5: stock_prices DB access layer
│   ├── ml/
│   │   └── dataset_builder.py     # Phase 6: ML dataset + label generation
│   ├── model/
│   │   ├── trainer.py             # Phase 7: ModelTrainer — load → train → eval → save
│   │   ├── predictor.py           # Phase 7: ModelPredictor — CSV / DataFrame / vector
│   │   ├── evaluator.py           # Phase 7: metric orchestration + JSON export
│   │   ├── metrics.py             # Accuracy, precision, recall, F1, confusion matrix
│   │   ├── model_io.py            # joblib save / load bundle
│   │   └── feature_importance.py  # Top-20 bar chart + ranked CSV
│   └── utils/
│       ├── config.py              # Pydantic settings (env-based)
│       ├── logger.py              # Structured logging
│       └── rate_limiter.py        # Token bucket rate limiter
│
├── tests/
│   ├── conftest.py                # Shared fixtures (settings mock)
│   └── unit/
│       ├── test_news_client.py
│       ├── test_gdelt_client.py           # GDELT client unit tests
│       ├── test_sentiment_analyzer.py
│       ├── test_run_sentiment.py          # Phase 2 script tests
│       ├── test_repository.py             # Phase 3: SQLite in-memory tests
│       ├── test_feature_engineer.py       # Phase 4: 95 tests (sentiment/rolling)
│       ├── test_technical_indicators.py   # Phase 4: 108 tests (technical indicators)
│       ├── test_price_client.py           # Phase 5: 62 tests (yfinance mocked)
│       ├── test_price_repository.py       # Phase 5: 60 tests (SQLite in-memory)
│       ├── test_dataset_builder.py        # Phase 6: 113 tests (SQLite in-memory)
│       ├── test_metrics.py                # Phase 7: metrics unit tests
│       ├── test_model_io.py               # Phase 7: model save/load tests
│       ├── test_predictor.py              # Phase 7: inference tests
│       └── test_trainer.py                # Phase 7: training pipeline tests
│
├── artifacts/
│   ├── models/
│   │   └── xgboost_direction_model.joblib # Trained model bundle
│   ├── metrics/
│   │   └── xgboost_metrics.json           # Evaluation metrics
│   └── plots/
│       ├── feature_importance.png          # Top-20 feature bar chart
│       └── feature_importance.csv          # All features ranked by importance
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

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env: set FINNHUB_API_KEY and DATABASE_URL
```

### 3. Run Phase 1 — Fetch news

**Option A: GDELT (historical backfill — recommended for ML training)**

```bash
# Backfill a full year for all tickers
python scripts/fetch_gdelt_news.py --start-date 2025-01-01 --end-date 2025-12-31

# Specific tickers and date range
python scripts/fetch_gdelt_news.py --tickers AAPL TSLA NVDA \
    --start-date 2025-01-01 --end-date 2025-06-30
```

Output: `data/raw/gdelt/<TICKER>_gdelt_<START>_<END>.csv`

**Option B: Finnhub (real-time, last N days)**

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

Output: `data/raw/<TICKER>_news_<YYYY-MM-DD>.csv`

### 4. Run Phase 2 — Sentiment analysis

**GDELT backfill files (direct file mode):**

```bash
python scripts/run_sentiment.py --input-file data/raw/gdelt/NVDA_gdelt_2025-01-01_2025-12-31.csv
```

**Finnhub files (date-based discovery):**

```bash
python scripts/run_sentiment.py
python scripts/run_sentiment.py --date 2026-06-15
```

The `--input-file` flag processes any single CSV directly — ticker and output tag are inferred from the filename. For GDELT files the output tag is the start year (e.g. `NVDA_sentiment_2025.csv`).

Options:

| Flag                       | Default            | Description                                     |
| -------------------------- | ------------------ | ----------------------------------------------- |
| `--tickers AAPL TSLA`      | all CSVs for today | Tickers to process (discovery mode)             |
| `--date 2026-06-15`        | today              | Date tag of input CSVs (discovery mode)         |
| `--input-file PATH`        | —                  | Process a single CSV directly (skips discovery) |
| `--model ProsusAI/finbert` | from `.env`        | Hugging Face model ID                           |
| `--batch-size 32`          | from `.env`        | Inference batch size                            |
| `--device auto`            | from `.env`        | `auto` / `cpu` / `cuda` / `mps`                 |
| `--log-level INFO`         | from `.env`        | Verbosity                                       |
| `--dry-run`                | —                  | Print config and exit                           |

Output: `data/processed/<TICKER>_sentiment_<tag>.csv`

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

| Flag                  | Default           | Description                             |
| --------------------- | ----------------- | --------------------------------------- |
| `--tickers AAPL TSLA` | all tickers in DB | Tickers to generate features for        |
| `--date 2026-06-16`   | today             | Target date (single-day mode)           |
| `--start-date`        | —                 | Start of date range (range mode)        |
| `--end-date`          | —                 | End of date range (range mode)          |
| `--output-dir PATH`   | `data/features/`  | Directory for output CSV                |
| `--lookback-days 7`   | `7`               | History window for rolling features     |
| `--log-level INFO`    | from `.env`       | Verbosity                               |
| `--dry-run`           | —                 | Print config and exit                   |

Output: `data/features/feature_dataset_<YYYY-MM-DD>.csv` (single day) or `data/features/feature_dataset_<START>_<END>.csv` (range)

### 7. Run Phase 5 — Stock price ingestion

```bash
# Create the stock_prices table and populate with one year of history
python scripts/fetch_prices.py --create-tables --lookback-days 365

# Specific tickers with a fixed date range
python scripts/fetch_prices.py --tickers AAPL TSLA NVDA \
    --start-date 2025-01-01 --end-date 2026-01-01

# Dry-run: fetch from Yahoo Finance but skip all database writes
python scripts/fetch_prices.py --tickers AAPL --lookback-days 30 --dry-run
```

Options:

| Flag                        | Default          | Description                                      |
| --------------------------- | ---------------- | ------------------------------------------------ |
| `--tickers AAPL TSLA`       | from `.env`      | Ticker symbols to fetch                          |
| `--start-date 2025-01-01`   | today − lookback | Inclusive start date                             |
| `--end-date 2026-01-01`     | today            | End date (exclusive per yfinance convention)     |
| `--lookback-days 365`       | `365`            | Days of history when `--start-date` is omitted  |
| `--create-tables`           | —                | Run `CREATE TABLE IF NOT EXISTS` before fetching |
| `--dry-run`                 | —                | Fetch data but skip all DB writes                |
| `--log-level INFO`          | from `.env`      | Verbosity                                        |

Output: rows upserted into the `stock_prices` table.

### 8. Run Phase 6 — Build ML dataset

```bash
# Specific date
python scripts/build_ml_dataset.py --date 2026-06-16

# Historical range
python scripts/build_ml_dataset.py \
    --features-path data/features/feature_dataset_2025-01-01_2026-06-17.csv

# Dry-run: print config and exit
python scripts/build_ml_dataset.py --dry-run
```

Output: `data/ml/ml_dataset_<date>.csv`

### 9. Run Phase 7 — Train model

```bash
# Auto-detect latest ML dataset in data/ml/
python scripts/train_model.py

# Explicit dataset
python scripts/train_model.py --dataset data/ml/ml_dataset_2025-01-01_2026-06-17.csv

# Custom artifact paths
python scripts/train_model.py \
    --model-out   artifacts/models/my_model.joblib \
    --metrics-out artifacts/metrics/my_metrics.json \
    --importance-out artifacts/plots/my_importance.png

# Dry-run (print config, no training)
python scripts/train_model.py --dry-run
```

Output: model artifact, metrics JSON, and feature importance chart in `artifacts/`.

### 10. Run predictions

```bash
# Predict on a dataset CSV
python scripts/predict.py \
    --input data/ml/ml_dataset_2026-06-16.csv

# Specify model and output paths
python scripts/predict.py \
    --model  artifacts/models/xgboost_direction_model.joblib \
    --input  data/ml/ml_dataset_2026-06-16.csv \
    --output data/ml/ml_dataset_2026-06-16_predictions.csv
```

---

## Historical News Backfill (GDELT)

The Finnhub free tier limits lookback to approximately one year and caps daily request volume. For meaningful ML training, the model needs multiple years of labelled examples per ticker.

**GDELT** (Global Database of Events, Language, and Tone) is a free, open, real-time database of global news events. Its news-article index covers hundreds of thousands of financial news sources and extends back many years.

The GDELT integration (`src/ingestion/gdelt_client.py`) enables:

- **Multi-year historical backfills** — fetch news for any date range, from months to years
- **Scalable ingestion** — rate-limited, batched requests with automatic retry
- **Identical downstream processing** — GDELT output uses the exact same column schema as Finnhub, so the same FinBERT pipeline, database loader, and feature engineer all work unchanged
- **Larger ML datasets** — more training samples lead to better generalisation and more reliable evaluation
- **Reproducible experiments** — fixed date ranges produce fully deterministic datasets

Both Finnhub and GDELT files feed into the same Phase 2 → 3 → 4 → 5 → 6 pipeline without any modification to the downstream code.

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
Yahoo Finance ─────────────── Phase 5: fetch_prices.py
(OHLCV prices)                           │  ▼
                                         │  stock_prices (PostgreSQL)
                                         │  │
                                         │  │  Phase 4: generate_features.py
                                         ▼  ▼  (sentiment + 19 technical indicators)
                              data/features/feature_dataset_<date>.csv
                                         │
                                         │  Phase 6: build_ml_dataset.py
                                         ▼
                              data/ml/ml_dataset_<date>.csv
                                         │
                                         │  Phase 7: train_model.py (XGBoost)
                                         ▼
                                   artifacts/ (model + metrics + plots)
                                         │
                                         │  predict.py
                                         ▼
                                   BUY / HOLD / SELL predictions
```

---

## Data Schemas

### Phase 1 — Raw news (`data/raw/<TICKER>_news_<date>.csv` or `data/raw/gdelt/<TICKER>_gdelt_<START>_<END>.csv`)

| Column         | Type           | Description                          |
| -------------- | -------------- | ------------------------------------ |
| `ticker`       | str            | Ticker symbol                        |
| `source_id`    | str            | Article ID (Finnhub) or hash (GDELT) |
| `source_name`  | str            | Publisher name                       |
| `title`        | str            | Article headline                     |
| `description`  | str            | Article summary                      |
| `url`          | str            | Article URL                          |
| `published_at` | datetime (UTC) | Publication timestamp                |
| `fetched_at`   | datetime (UTC) | Fetch timestamp                      |

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
          │                                        stock_prices
          │                                              │
          │                                   load_price_data()
          │                                              │
          │  generate_features()  ◄──────────────────────┘
          │  (per-ticker: sentiment + technical indicators)
          ▼
  features_df  (one row per ticker — 43 columns)
          │
          │  save_features()
          ▼
  data/features/feature_dataset_<date>.csv
```

**Historical backfill:** `generate_features.py` supports a `--start-date` / `--end-date` range mode that iterates over every date in the window, producing a single combined CSV covering the full period.

**Technical indicators** are derived from historical OHLCV data stored in `stock_prices` and computed with **pandas only** — no external TA libraries. A 90-calendar-day price lookback window (~63 trading days) is loaded automatically, providing sufficient warm-up for every indicator including the slowest (MACD signal line, ~35 trading days). When price data is unavailable, technical columns are `None` rather than arbitrary fill values.

#### Exception hierarchy

```
FeatureEngineeringError          (base)
├── DataLoadError                (database connectivity / query failure)
└── FeatureGenerationError       (empty input / no articles on target date)
```

### Feature Definitions

Each row in the output dataset represents one ticker on one date. The dataset contains **41 engineered features** across six groups (plus `ticker` and `date` = 43 columns total).

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
| `mean_sentiment_score` | Mean of `sentiment_score` (−1 / 0 / +1) on target date         |
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

#### Rolling Sentiment Features (4)

Rolling means are computed from **daily aggregates** — each calendar day contributes one data point (its daily mean sentiment) regardless of article volume, giving each day equal weight.

| Feature                     | Window | Description                                    |
| --------------------------- | ------ | ---------------------------------------------- |
| `rolling_3d_mean_sentiment` | 3 days | Mean of the last 3 daily mean sentiment scores |
| `rolling_7d_mean_sentiment` | 7 days | Mean of the last 7 daily mean sentiment scores |
| `rolling_3d_article_volume` | 3 days | Total article count in the last 3 days         |
| `rolling_7d_article_volume` | 7 days | Total article count in the last 7 days         |

#### Trend Indicators (4)

Derived from `close_price` in `stock_prices` using pandas `rolling()` (SMA) and `ewm(adjust=False)` (EMA).

| Feature  | Description                                |
| -------- | ------------------------------------------ |
| `sma_10` | Simple Moving Average over 10 trading days |
| `sma_20` | Simple Moving Average over 20 trading days |
| `ema_10` | Exponential Moving Average, span = 10      |
| `ema_20` | Exponential Moving Average, span = 20      |

#### Momentum Indicators (4)

| Feature          | Description                                                        |
| ---------------- | ------------------------------------------------------------------ |
| `rsi_14`         | RSI (14) using Wilder's EWM smoothing (alpha = 1/14); range [0, 100] |
| `macd`           | MACD line = EMA(12) − EMA(26)                                      |
| `macd_signal`    | Signal line = EMA(9) of MACD                                       |
| `macd_histogram` | Histogram = MACD line − signal line                                |

#### Volatility Indicators (5)

| Feature          | Description                                                                         |
| ---------------- | ----------------------------------------------------------------------------------- |
| `bb_upper`       | Bollinger upper band: SMA(20) + 2 × rolling std                                     |
| `bb_lower`       | Bollinger lower band: SMA(20) − 2 × rolling std                                     |
| `bb_width`       | Bandwidth = (upper − lower) / SMA(20)                                               |
| `atr_14`         | Average True Range (14) via Wilder's EWM; TR = max(H−L, \|H−C₋₁\|, \|L−C₋₁\|)    |
| `volatility_20d` | 20-day rolling std of daily percentage returns (sample std, ddof = 1)               |

#### Price Return Features (3)

Column names use the `price_chg_` prefix to avoid collision with the `return_*` ML label columns produced by Phase 6.

| Feature         | Description                                               |
| --------------- | --------------------------------------------------------- |
| `price_chg_1d`  | 1-trading-day percentage price change: `(P₀ − P₋₁) / P₋₁` |
| `price_chg_5d`  | 5-trading-day percentage price change                      |
| `price_chg_10d` | 10-trading-day percentage price change                     |

#### Volume Features (3)

| Feature             | Description                                                    |
| ------------------- | -------------------------------------------------------------- |
| `volume_change_pct` | 1-day percentage change in trading volume                      |
| `volume_avg_5d`     | 5-day rolling mean of volume (`min_periods = 5`)               |
| `volume_ratio`      | Today's volume divided by `volume_avg_5d` (relative activity)  |

### Missing Value Policy

The first rows of any price series will naturally have `NaN` for indicators that require N periods of warm-up (e.g. `sma_20` needs 20 rows). When the target date falls within such a warm-up period, the indicator value is `None` in the output CSV rather than an arbitrary fill. Downstream stages receive these as `NaN` and handle them through normal ML preprocessing.

---

## Phase 6 — ML Dataset Builder

Phase 6 assembles the supervised training dataset by joining Phase 4 feature vectors with future stock price movements to produce binary and multi-class labels for every (ticker, date) pair.

Running against the full historical feature dataset (579 rows, Jan 2025 – Jun 2026) produced **488 labelled ML samples** — the remainder were dropped because insufficient forward price data was available at the end of the date range.

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

### Label Generation Workflow

For each `(ticker, date)` row in the feature dataset:

1. **Locate today's close** — look up `close_price` for `date` in `stock_prices`.
2. **Find future closes** — using a trading-day index (not calendar days), find the closing price N trading days ahead for N ∈ {1, 3, 5, 7}.
3. **Compute returns** — `return_Nd = (future_close - close_today) / close_today`
4. **Assign binary labels** — `1` if return > 0, `0` otherwise.
5. **Assign direction label** — `BUY` / `HOLD` / `SELL` from the 5-day return.

Rows for which any required future close is unavailable are **logged and skipped** — they do not appear in the output.

### BUY / HOLD / SELL Definitions

| Label  | Condition           |
| ------ | ------------------- |
| `BUY`  | `return_5d > 0.02`  |
| `SELL` | `return_5d < -0.02` |
| `HOLD` | otherwise           |

> **Trading-day indexing** — the "Nth trading day ahead" skips weekends and market holidays automatically because only dates present in `stock_prices` are considered.

### ML Dataset Schema

The output CSV contains **all 41 Phase 4 feature columns** (plus `ticker` and `date`) followed by **13 label columns** (56 columns total).

#### Label Columns (13)

| Column             | Type    | Description                                         |
| ------------------ | ------- | --------------------------------------------------- |
| `future_close_1d`  | float   | Closing price 1 trading day after target date       |
| `future_close_3d`  | float   | Closing price 3 trading days after target date      |
| `future_close_5d`  | float   | Closing price 5 trading days after target date      |
| `future_close_7d`  | float   | Closing price 7 trading days after target date      |
| `return_1d`        | float   | `(future_close_1d − close_today) / close_today`     |
| `return_3d`        | float   | `(future_close_3d − close_today) / close_today`     |
| `return_5d`        | float   | `(future_close_5d − close_today) / close_today`     |
| `return_7d`        | float   | `(future_close_7d − close_today) / close_today`     |
| `label_up_1d`      | int 0/1 | 1 if `return_1d > 0`                                |
| `label_up_3d`      | int 0/1 | 1 if `return_3d > 0`                                |
| `label_up_5d`      | int 0/1 | 1 if `return_5d > 0`                                |
| `label_up_7d`      | int 0/1 | 1 if `return_7d > 0`                                |
| `label_direction`  | str     | `BUY` / `HOLD` / `SELL` (derived from `return_5d`) |

### Missing Data Handling

| Scenario                             | Behaviour                           |
| ------------------------------------ | ----------------------------------- |
| Ticker not in `stock_prices`         | Row skipped, warning logged         |
| Feature date not in `stock_prices`   | Row skipped, warning logged         |
| Insufficient future trading days     | Row skipped, warning logged         |
| All required future closes missing   | `LabelGenerationError` raised       |
| `NULL` close price in database       | Treated as missing, row skipped     |

---

## Phase 7 — XGBoost Model Training & Prediction

Phase 7 trains a supervised XGBoost classifier on the Phase 6 ML dataset to predict whether a stock will go **BUY**, **HOLD**, or **SELL** over the next 5 trading days.

### Package — `src/model/`

| Module                  | Purpose                                                              |
| ----------------------- | -------------------------------------------------------------------- |
| `trainer.py`            | `ModelTrainer` — full training pipeline (load → train → eval → save) |
| `predictor.py`          | `ModelPredictor` — inference from CSV, DataFrame, or vector          |
| `evaluator.py`          | `ModelEvaluator` — metric orchestration + logging + JSON export      |
| `metrics.py`            | Accuracy, precision, recall, F1, confusion matrix, report            |
| `model_io.py`           | `save_model()` / `load_model()` via joblib                          |
| `feature_importance.py` | Importance computation, top-20 bar chart, ranked CSV                 |

### Artifact Outputs

| Artifact                                          | Description                                             |
| ------------------------------------------------- | ------------------------------------------------------- |
| `artifacts/models/xgboost_direction_model.joblib` | Serialised model bundle (model + encoder + feature list) |
| `artifacts/metrics/xgboost_metrics.json`          | Full evaluation metrics (accuracy, F1, confusion matrix) |
| `artifacts/plots/feature_importance.png`          | Horizontal bar chart of top-20 features by importance    |
| `artifacts/plots/feature_importance.csv`          | All features ranked by importance score (descending)     |

### Model Configuration

| Parameter          | Value              | Notes                                     |
| ------------------ | ------------------ | ----------------------------------------- |
| Estimator          | `XGBClassifier`    | sklearn-compatible API                    |
| Objective          | `multi:softprob`   | Multi-class with probability output       |
| Classes            | BUY / HOLD / SELL  | Encoded 0 / 1 / 2 via `LabelEncoder`     |
| `n_estimators`     | 300                | Number of boosting rounds                 |
| `max_depth`        | 6                  | Maximum tree depth                        |
| `learning_rate`    | 0.05               | Shrinkage per step                        |
| `subsample`        | 0.8                | Row subsampling ratio                     |
| `colsample_bytree` | 0.8                | Column subsampling per tree               |
| Train/test split   | 80 / 20            | Stratified, `random_state=42`             |

### Feature Detection

Before training, non-feature columns are automatically excluded:

- **Metadata**: `ticker`, `date`
- **Future-close labels**: `future_close_*`
- **Return labels**: `return_*`
- **Binary / direction labels**: `label_*`

All remaining numeric columns (41 engineered features) are used for training.

### Training Summary

```
============================================================
  Phase 7: XGBoost Model Training
============================================================
  Dataset rows    : 488
  Feature columns : 41
  Train rows      : 378
  Test rows       : 95
  Accuracy        : 0.5158
  Macro F1        : 0.5141
  Macro Precision : 0.5217
  Macro Recall    : 0.5145
  Model saved     : artifacts/models/xgboost_direction_model.joblib
============================================================
```

### Prediction Output

The output CSV contains all original columns plus four new columns:

| Column                | Type  | Description                     |
| --------------------- | ----- | ------------------------------- |
| `predicted_direction` | str   | `BUY`, `HOLD`, or `SELL`        |
| `prob_BUY`            | float | Probability score for BUY class  |
| `prob_HOLD`           | float | Probability score for HOLD class |
| `prob_SELL`           | float | Probability score for SELL class |

### Python API

```python
from src.model.trainer import ModelTrainer
from src.model.predictor import ModelPredictor
from pathlib import Path

# ── Training ──────────────────────────────────────────────
trainer = ModelTrainer(
    dataset_path=Path("data/ml/ml_dataset_2025-01-01_2026-06-17.csv"),
    model_out=Path("artifacts/models/xgboost_direction_model.joblib"),
    metrics_out=Path("artifacts/metrics/xgboost_metrics.json"),
    importance_out=Path("artifacts/plots/feature_importance.png"),
)
trainer.load_dataset().prepare_features().train().evaluate()
trainer.save_model()

# ── Inference from CSV ────────────────────────────────────
predictor = ModelPredictor(
    model_path=Path("artifacts/models/xgboost_direction_model.joblib")
)
predictor.load_model()
result_df = predictor.predict_from_csv(Path("data/ml/ml_dataset_2026-06-16.csv"))

# ── Inference from a single feature vector ────────────────
result = predictor.predict_from_vector({
    "article_count":  8.0,
    "positive_ratio": 0.62,
    "sma_10":         192.34,
    "rsi_14":         58.21,
    # … all 41 feature columns …
})
print(result["predicted_direction"])   # "BUY"
print(result["probabilities"])         # {"BUY": 0.58, "HOLD": 0.27, "SELL": 0.15}
```

### Metrics JSON Schema

```json
{
  "accuracy": 0.5158,
  "precision": {
    "macro": 0.5217,
    "per_class": { "BUY": 0.54, "HOLD": 0.52, "SELL": 0.50 }
  },
  "recall": {
    "macro": 0.5145,
    "per_class": { "BUY": 0.52, "HOLD": 0.54, "SELL": 0.49 }
  },
  "f1": {
    "macro": 0.5141,
    "per_class": { "BUY": 0.53, "HOLD": 0.53, "SELL": 0.50 }
  },
  "confusion_matrix": [[...], [...], [...]],
  "labels": ["BUY", "HOLD", "SELL"],
  "classification_report": "..."
}
```

---

## Configuration

All settings are managed via environment variables (or `.env`):

| Variable             | Default                    | Description                                |
| -------------------- | -------------------------- | ------------------------------------------ |
| `FINNHUB_API_KEY`    | **required**               | Finnhub API key                            |
| `TICKERS`            | `AAPL,TSLA,NVDA,MSFT,AMZN` | Default ticker list                        |
| `NEWS_LOOKBACK_DAYS` | `7`                        | Days of news history to fetch (Finnhub)    |
| `LOG_LEVEL`          | `INFO`                     | Logging verbosity                          |
| `FINBERT_MODEL`      | `ProsusAI/finbert`         | Hugging Face model ID                      |
| `FINBERT_BATCH_SIZE` | `32`                       | Inference batch size                       |
| `FINBERT_DEVICE`     | `auto`                     | Compute device (`auto`/`cpu`/`cuda`/`mps`) |
| `DATABASE_URL`       | `None`                     | PostgreSQL connection URL (Phase 3+)       |

---

## Database Setup

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

Indexes on stock_prices:
  ix_stock_prices_ticker       on ticker
  ix_stock_prices_ticker_date  on (ticker, trading_date)
```

Re-running any load script is always safe — all tables use `ON CONFLICT DO UPDATE` (PostgreSQL) or SELECT-then-UPDATE (SQLite/tests), so existing rows are refreshed rather than duplicated.

---

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest                          # all tests
pytest -v                       # verbose
pytest tests/unit/test_technical_indicators.py   # technical indicator tests only
pytest --cov=src --cov-report=term-missing       # with coverage
```

All unit tests mock the Hugging Face pipeline and use an in-memory SQLite database — no model download and no PostgreSQL instance are needed.

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

## Future Work

| Area                         | Description                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------- |
| **Hyperparameter tuning**    | Grid search or Bayesian optimisation for XGBoost parameters                      |
| **Time-series cross-validation** | Walk-forward validation to prevent data leakage across time                  |
| **SHAP explainability**      | Per-prediction feature attribution using SHAP values                             |
| **Real-time inference**      | Live pipeline connecting Finnhub stream → FinBERT → feature engineering → model  |
| **Streamlit dashboard**      | Interactive UI for signal monitoring, feature exploration, and prediction history |
| **Docker deployment**        | Containerised pipeline with `docker-compose` for one-command setup               |
| **Cloud deployment**         | Scheduled execution on AWS / GCP with managed PostgreSQL                         |
| **Additional data sources**  | Earnings call transcripts, SEC filings, options flow, macroeconomic indicators   |
