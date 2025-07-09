import os
import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DB_URI")
engine = create_engine(DATABASE_URL)

def load_stock_data(symbol: str):
    df = yf.download(symbol, period="1d", interval="1d")

    if df.empty:
        raise ValueError(f"No data returned for symbol {symbol}")

    df.reset_index(inplace=True)
    df["symbol"] = symbol.upper()
    df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    }, inplace=True)

    with engine.begin() as conn:
        # Create schema and table if needed
        conn.execute(text("""
            # CREATE SCHEMA IF NOT EXISTS analytics;
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
        # Insert
        df.to_sql("stock_prices", con=conn, schema="analytics", if_exists="append", index=False)
        print(f"✅ Successfully loaded data for {symbol}")
