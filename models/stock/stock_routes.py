from flask import Blueprint, request, redirect, url_for, flash, render_template
from sqlalchemy import text
from __init__ import db
# from main import db  # ✅ Use actual app file name here (e.g., main.py)
from models.stock.etl import load_stock_data

# ✅ Create Blueprint
bp_stock = Blueprint("bp_stock", __name__)

# ✅ Route: POST → Manually run ETL from form
@bp_stock.route("/run_etl", methods=["POST"])
def run_etl():
    symbol = request.form.get("symbol", "TSLA").upper()
    try:
        load_stock_data(symbol)
        flash(f"✅ Successfully loaded {symbol}")
    except Exception as e:
        flash(f"❌ ETL failed: {str(e)}")
    return redirect(url_for("bp_stock.trigger_etl_form"))

# ✅ Route: GET → Show latest 20 records
@bp_stock.route("/stock_data", methods=["GET"])
def stock_data():
    try:
        result = db.session.execute(
            text("SELECT * FROM analytics.stock_prices ORDER BY date DESC LIMIT 20")
        )
        rows = result.fetchall()
    except Exception as e:
        flash(f"❌ Failed to fetch stock data: {str(e)}")
        rows = []
    return render_template("stock_data.html", rows=rows)

# ✅ Route: GET → Show ETL trigger form
@bp_stock.route("/trigger_etl", methods=["GET"])
def trigger_etl_form():
    return render_template("trigger_etl.html")

# ✅ Route: GET → Show basic 30-day line chart
@bp_stock.route("/stocks/<symbol>", methods=["GET"])
def stock_chart(symbol):
    try:
        result = db.session.execute(text("""
            SELECT date, close FROM analytics.stock_prices
            WHERE symbol = :symbol
            ORDER BY date ASC
            LIMIT 30
        """), {"symbol": symbol.upper()})
        rows = result.fetchall()
        dates = [row.date.strftime("%Y-%m-%d") for row in rows]
        closes = [row.close for row in rows]
    except Exception as e:
        flash(f"❌ Failed to fetch chart data for {symbol}: {str(e)}")
        dates, closes = [], []
    return render_template("stock_chart.html", symbol=symbol.upper(), dates=dates, closes=closes)
