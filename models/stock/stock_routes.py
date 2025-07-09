from flask import Blueprint, request, redirect, url_for, flash, render_template
from sqlalchemy import text
from models import db
from models.stock.etl import load_stock_data  # ✅ Correct import

# ✅ Create Blueprint
bp_stock = Blueprint("bp_stock", __name__)

# ✅ POST route to manually trigger ETL from browser form
@bp_stock.route("/run_etl", methods=["POST"])
def run_etl():
    symbol = request.form.get("symbol", "AAPL").upper()

    try:
        load_stock_data(symbol)
        flash(f"✅ Successfully loaded {symbol}")
    except Exception as e:
        flash(f"❌ ETL failed: {str(e)}")

    return redirect(url_for("get_all_posts"))  # Redirect to blog homepage


# ✅ GET route to preview the most recent 20 stock prices from DB
@bp_stock.route("/stock_data")
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
