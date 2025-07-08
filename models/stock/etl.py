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

# ✅ Get DB URI from .env
DATABASE_URL = os.getenv("DB_URI")
if not DATABASE_URL:
    raise ValueError("❌ Missing DB_URI in .env or .env.local")

# ✅ Connect to PostgreSQL
engine = create_engine(DATABASE_URL)

# ✅ Ensure schema/table exist
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
    print("✅ analytics.stock_prices table is ready")

# ✅ ETL function
def load_stock_data(symbol="AAPL", table_name="stock_prices"):
    print(f"\n📥 Fetching data for {symbol}")
    df = yf.download(symbol, period="7d", interval="1d")

    if df.empty:
        print(f"⚠️ No data found for {symbol}")
        return

    # ✅ Clean MultiIndex if exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # ✅ Rename columns
    df.reset_index(inplace=True)
    df.columns.name = None
    df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    }, inplace=True)

    df["symbol"] = symbol
    df = df[["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]]

    # ✅ Sanity check
    print(f"🧪 Columns: {df.columns.tolist()}")
    print(df.head())

    # ✅ Upload to PostgreSQL
    df.to_sql(table_name, con=engine, schema="analytics", index=False, if_exists="append")
    print("✅ Insert complete for", symbol)

# ✅ Run manually
if __name__ == "__main__":
    load_stock_data("AAPL")

