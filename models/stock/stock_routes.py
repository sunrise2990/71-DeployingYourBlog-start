from flask import Blueprint, request, redirect, url_for, flash, render_template, jsonify
from sqlalchemy import text
from models import db
from models.stock.etl import load_stock_data
import logging

bp_stock = Blueprint("bp_stock", __name__)
logger = logging.getLogger(__name__)

# ✅ Route: POST → Trigger ETL via form
@bp_stock.route("/run_etl", methods=["POST"])
def run_etl():
    symbol = request.form.get("symbol", "TSLA").upper()
    try:
        load_stock_data(symbol)
        flash(f"✅ Successfully loaded {symbol}")
    except Exception as e:
        logger.error(f"ETL failed for {symbol}: {e}")
        flash(f"❌ ETL failed: {str(e)}")
    return redirect(url_for("bp_stock.trigger_etl_form"))

# ✅ Route: GET → Display 20 most recent rows
@bp_stock.route("/stock_data", methods=["GET"])
def stock_data():
    try:
        result = db.session.execute(
            text("SELECT * FROM analytics.stock_prices ORDER BY date DESC LIMIT 20")
        )
        rows = result.fetchall()
        if not rows:
            flash("⚠️ No data found in analytics.stock_prices.")
        else:
            flash(f"✅ Pulled {len(rows)} records from stock_prices.")
    except Exception as e:
        print("❌ ERROR in /stock_data:", str(e))  # 👈 Log the error to terminal
        flash(f"❌ Failed to fetch stock data: {str(e)}")
        rows = []
    return render_template("stock_data.html", rows=rows)

# ✅ Route: GET → JSON API to debug stock data
@bp_stock.route("/api/stock_data", methods=["GET"])
def stock_data_json():
    try:
        result = db.session.execute(
            text("SELECT * FROM analytics.stock_prices ORDER BY date DESC LIMIT 20")
        )
        rows = result.fetchall()
        data = [dict(row._mapping) for row in rows]
        return jsonify(data)
    except Exception as e:
        logger.error(f"/api/stock_data error: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Route: GET → Trigger ETL form
@bp_stock.route("/trigger_etl", methods=["GET"])
def trigger_etl_form():
    return render_template("trigger_etl.html")

# ✅ Route: GET → Line chart for latest 30 closes
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
        return render_template("stock_chart.html", symbol=symbol.upper(), dates=dates, closes=closes)
    except Exception as e:
        logger.error(f"Chart data fetch failed for {symbol}: {e}")
        flash(f"❌ Failed to fetch chart data for {symbol}: {str(e)}")
        return render_template("stock_chart.html", symbol=symbol.upper(), dates=[], closes=[]), 500
