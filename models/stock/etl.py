import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
from dotenv import load_dotenv
from pathlib import Path

# Dynamically load .env.local if exists, else fallback to .env
if Path(".env.local").exists():
    load_dotenv(dotenv_path=".env.local")
else:
    load_dotenv()

# Fetch DB URI from env
DATABASE_URL = os.getenv("DB_URI")

engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    result = conn.execute("SELECT NOW()")
    print("PostgreSQL connected:", result.scalar())

# Raise error if not found
if not DATABASE_URL:
    raise ValueError("❌ DATABASE_URL is missing from .env")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

def fetch_and_store_stock(symbol="AAPL", table_name="stock_prices"):
    print(f"📥 Downloading {symbol} from Yahoo Finance...")
    df = yf.download(symbol, start="2024-01-01", end="2025-01-01")
    df.reset_index(inplace=True)
    df["symbol"] = symbol

    print("🔍 Sample data:")
    print(df.head())

    print(f"🗃 Writing to PostgreSQL → schema=analytics, table={table_name}")
    df.to_sql(
        table_name,
        engine,
        schema="analytics",  # ✅ Your schema
        if_exists="replace",
        index=False
    )

    print("✅ Upload complete!")

# Run if executed directly
if __name__ == "__main__":
    fetch_and_store_stock()
