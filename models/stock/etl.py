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
    raise ValueError("Missing DB_URI in .env")

# ✅ Connect to PostgreSQL
engine = create_engine(DATABASE_URL)

# ✅ Ensure schema and clean table
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
    print("✅ analytics.stock_prices ready")

# ✅ Main ETL Function
def load_stock_data(symbol="AAPL", table_name="stock_prices"):
    print(f"📥 Fetching data for: {symbol}")
    df = yf.download(symbol, period="1d", interval="1d")

    if df.empty:
        print(f"⚠️ No data for {symbol}")
        return

    # ✅ Flatten MultiIndex if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    # ✅ Reset index and rename
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

    print(f"🧪 Cleaned Columns: {df.columns.tolist()}")
    print(df.head())

    # ✅ Upload to PostgreSQL
    df.to_sql(table_name, con=engine, schema="analytics", index=False, if_exists="append")
    print("✅ Insert complete")

# ✅ Run directly if needed
if __name__ == "__main__":
    load_stock_data("AAPL")
