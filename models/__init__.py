from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile('.env.local')

    db.init_app(app)

    from models.stock.stock_routes import bp_stock
    app.register_blueprint(bp_stock)

    return app
