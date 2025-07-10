# models/__init__.py
from sqlalchemy import create_engine
from flask_sqlalchemy import SQLAlchemy
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DB_URI")

# ✅ This should be declared here
engine = create_engine(DATABASE_URL)

db = SQLAlchemy()  # ✅ Add this
