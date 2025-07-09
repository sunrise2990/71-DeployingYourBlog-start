import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

# ✅ Load .env.local or fallback to default .env
if Path(".env.local").exists():
    load_dotenv(".env.local")
else:
    load_dotenv()

# ✅ Get the PostgreSQL connection URL from environment
DATABASE_URL = os.getenv("DB_URI")
if not DATABASE_URL:
    raise ValueError("Missing DB_URI in .env")

# ✅ Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# ✅ Function to create schema and table — ONLY call manually if needed
def create_stock_table():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE SCHEMA IF NOT EXISTS analytics;

            DROP TABLE IF EXISTS analytics.stock_prices;

            CREATE TABLE analytics.stock_prices (
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
        print("✅ Table analytics.stock_prices is ready")

# ✅ Function to fetch and store stock data
def load_stock_data(symbol="AAPL", table_name="stock_prices"):
    print(f"🔄 Fetching {symbol}")
    df = yf.download(symbol, period="1d", interval="1d")

    if df.empty:
        print("⚠️ No data returned")
        return

    # ✅ Clean DataFrame columns
    df.reset_index(inplace=True)
    df.columns = [col.lower() for col in df.columns]
    df.rename(columns={"adj close": "adj_close"}, inplace=True)
    df["symbol"] = symbol

    print("📊 Cleaned columns:", df.columns.tolist())
    print(df.head())

    # ✅ Write to PostgreSQL
    df.to_sql(table_name, con=engine, schema="analytics", index=False, if_exists="append")
    print("✅ Inserted to DB")

# ✅ Optional: run manually for testing
if __name__ == "__main__":
    create_stock_table()     # ⚠️ Only run this once unless you want to drop the table
    load_stock_data("AAPL")
