# models/stock/stock_routes.py

from sqlalchemy import text
from flask import Blueprint, request, redirect, url_for, flash, render_template
from models.stock.etl import load_stock_data
from models import db  # ✅ this is correct

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
    try:
        result = db.session.execute(
            text('SELECT * FROM analytics.stock_prices ORDER BY "date" DESC LIMIT 20')
        )
        rows = result.fetchall()
        return render_template("stock_data.html", rows=rows)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)  # logs to terminal
        return f"<h2>❌ Error</h2><pre>{tb}</pre>", 500

