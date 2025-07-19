from main import app, db
from flask_migrate import Migrate
from flask.cli import main as flask_main

migrate = Migrate(app, db)

if __name__ == "__main__":
    flask_main()