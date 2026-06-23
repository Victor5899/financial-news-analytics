from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ingestion.news_client import fetch_articles
ticker = "AAPL"

df = fetch_articles(
    ticker="AAPL",
    from_date=datetime(2025, 3, 1),
    to_date=datetime(2025, 3, 15),
)

print(f"Rows returned: {len(df)}")

if not df.empty:
    print("\nEarliest:")
    print(df["published_at"].min())

    print("\nLatest:")
    print(df["published_at"].max())


    print("\nColumns:")
    print(df.columns.tolist())