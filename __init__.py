# 71-DeployingYourBlog-start/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def create_app():
    app = Flask(__name__)

    # Load config from .env
    app.config.from_pyfile('.env.local')

    # Initialize DB
    db.init_app(app)

    # Register Blueprints
    from models.stock.stock_routes import bp_stock
    app.register_blueprint(bp_stock)

    return app