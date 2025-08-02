import sys
import os
from dotenv import load_dotenv

# === Detect current project path ===
project_home = os.path.abspath(os.path.dirname(__file__))

if project_home not in sys.path:
    sys.path.insert(0, project_home)

# === Load environment variables ===
load_dotenv(os.path.join(project_home, '.env'))         # Load EC2 .env
load_dotenv(os.path.join(project_home, '.env.local'))   # Load local dev .env if exists

print("Loaded DB_URI:", os.getenv("DB_URI"))

# === Ensure Flask key (fallback if not found) ===
if not os.environ.get("FLASK_KEY"):
    os.environ["FLASK_KEY"] = os.environ.get("FLASK_KEY", "fallback-secret-key")

# === Launch the Flask app ===
from main import app as application

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000)



