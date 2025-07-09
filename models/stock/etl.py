import os
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pathlib import Path

# ✅ Load environment variables
if Path(".env.local").exists():
    load_dotenv(".env.local")
else:
    load_dotenv()

DATABASE_URL = os.getenv("DB_URI")
if not DATABASE_URL:
    raise ValueError("Missing DB_URI in .env")

# ✅ Connect to PostgreSQL
engine = create_engine(DATABASE_URL)

# ✅ Create schema and clean table once (remove this block later if stable)
with engine.begin() as conn:
    conn.execute(text("""
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

# ✅ Load data and insert
def load_stock_data(symbol="AAPL", table_name="stock_prices"):
    print(f"🔄 Fetching {symbol}")
    df = yf.download(symbol, period="1d", interval="1d")

    if df.empty:
        print("⚠️ No data returned")
        return

    # ✅ Clean columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    df.reset_index(inplace=True)
    df.columns = [col.lower() for col in df.columns]
    df.rename(columns={"adj close": "adj_close"}, inplace=True)
    df["symbol"] = symbol

    print("📊 Cleaned columns:", df.columns.tolist())
    print(df.head())

    # ✅ Upload to PostgreSQL
    df.to_sql(table_name, con=engine, schema="analytics", index=False, if_exists="append")
    print("✅ Inserted to DB")

if __name__ == "__main__":
    load_stock_data("AAPL")
