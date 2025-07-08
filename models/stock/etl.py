import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

if Path(".env.local").exists():
    load_dotenv(dotenv_path=".env.local")
else:
    load_dotenv()

DATABASE_URL = os.getenv("DB_URI")
if not DATABASE_URL:
    raise ValueError("Missing DB_URI in .env")

engine = create_engine(DATABASE_URL)

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
    print("✅ analytics.stock_prices ready")

def load_stock_data(symbol="AAPL", table_name="stock_prices"):
    import yfinance as yf
    import pandas as pd

    print(f"📥 Fetching data for: {symbol}")
    df = yf.download(symbol, period="1d", interval="1d")

    if df.empty:
        print(f"⚠️ No data for {symbol}")
        return

    # ✅ Flatten column MultiIndex IMMEDIATELY
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    # ✅ Rename columns to match DB table
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

    # ✅ Clean column names
    df.columns = [col.strip() for col in df.columns]
    print("🧪 Final Columns:", df.columns.tolist())
    print(df.head())

    # ✅ Upload to PostgreSQL
    df.to_sql(table_name, con=engine, schema="analytics", index=False, if_exists="append")
    print("✅ Insert complete")

if __name__ == "__main__":
    load_stock_data("AAPL")

