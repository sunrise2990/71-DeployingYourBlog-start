import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

# ✅ Load environment variables from .env or .env.local
if Path(".env.local").exists():
    load_dotenv(dotenv_path=".env.local")
else:
    load_dotenv()

# ✅ Read DB URI securely
DATABASE_URL = os.getenv("DB_URI")
if not DATABASE_URL:
    raise ValueError("❌ Missing DB_URI in .env file")

# ✅ Create PostgreSQL engine
engine = create_engine(DATABASE_URL)

# ✅ Ensure schema and table exist
with engine.begin() as conn:
    conn.execute(text("""
        CREATE SCHEMA IF NOT EXISTS analytics;

        CREATE TABLE IF NOT EXISTS analytics.stock_prices (
            id SERIAL PRIMARY KEY,
            symbol TEXT,
            date TIMESTAMP,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            adj_close FLOAT,
            volume BIGINT
        );
    """))
    print("✅ PostgreSQL table `analytics.stock_prices` is ready.")

# ✅ Main ETL function
def load_stock_data(symbol="AAPL", table_name="stock_prices"):
    print(f"📥 Fetching data for: {symbol}")
    df = yf.download(symbol, period="1d", interval="1d")
    if df.empty:
        print(f"⚠️ No data returned for {symbol}")
        return

    df.reset_index(inplace=True)
    df["symbol"] = symbol

    df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    }, inplace=True)

    print(f"🔍 Sample data:\n{df.head()}")
    print(f"🗃 Inserting into analytics.{table_name}...")

    df.to_sql(
        table_name,
        con=engine,
        schema="analytics",
        if_exists="append",
        index=False
    )

    print("✅ Upload complete.")

# ✅ If run standalone (test/dev)
if __name__ == "__main__":
    load_stock_data("AAPL")