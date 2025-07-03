import sys
import os
from dotenv import load_dotenv

# === Path Configuration ===
project_home = '/var/www/71-DeployingYourBlog-start'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# === Load .env if running locally (optional) ===
load_dotenv(os.path.join(project_home, '.env'))

# === Ensure Flask Key is loaded (required for CSRF) ===
if not os.environ.get("FLASK_KEY"):
    os.environ["FLASK_KEY"] = os.environ.get("FLASK_KEY", "fallback-secret-key")

# === Launch Flask App ===
from main import app as application




