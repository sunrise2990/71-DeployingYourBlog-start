# models/stock/stock_routes.py
from sqlalchemy import text
from flask import Blueprint, request, redirect, url_for, flash, render_template
from models.stock.etl import load_stock_data
from models import db

bp_stock = Blueprint("bp_stock", __name__)

@bp_stock.route("/run_etl", methods=["POST"])
def run_etl():
    symbol = request.form.get("symbol", "AAPL").upper()
    try:
        load_stock_data(symbol)
        flash(f"✅ Successfully loaded {symbol}")
    except Exception as e:
        flash(f"❌ ETL failed: {str(e)}")
    return redirect(url_for("index"))

@bp_stock.route("/stock_data")
def stock_data():
    result = db.session.execute(text("SELECT * FROM analytics.stock_prices ORDER BY date DESC LIMIT 20"))
    return render_template("stock_data.html", rows=result.fetchall())

