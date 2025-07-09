import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

# ✅ Load environment variables
if Path(".env.local").exists():
    load_dotenv(dotenv_path=".env.local")
else:
    load_dotenv()

# ✅ Get DB URI
DATABASE_URL = os.getenv("DB_URI")
if not DATABASE_URL:
    raise ValueError("❌ Missing DB_URI in .env or .env.local")

# ✅ Connect to PostgreSQL
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
    print("✅ Created analytics.stock_prices (if not exists)")


# ✅ ETL FUNCTION
def load_stock_data(symbol="AAPL", table_name="stock_prices"):
    print(f"\n📥 Downloading: {symbol}")
    df = yf.download(symbol, period="1d", interval="1d")

    if df.empty:
        print("⚠️ No data returned from Yahoo Finance")
        return

    # ✅ FLATTEN MultiIndex columns
    df.columns = [col if isinstance(col, str) else col[0] for col in df.columns]

    # ✅ Reset index to move date into column
    df.reset_index(inplace=True)

    # ✅ Rename columns
    df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    }, inplace=True)

    # ✅ Add symbol column
    df["symbol"] = symbol

    # ✅ Reorder columns to match DB table
    df = df[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]]

    print("📊 Prepared Data:")
    print(df.head())

    # ✅ Load to database
    df.to_sql(name=table_name, con=engine, schema="analytics", index=False, if_exists="append")
    print("✅ Inserted to analytics.stock_prices")


# ✅ Manual trigger
if __name__ == "__main__":
    load_stock_data("AAPL")
