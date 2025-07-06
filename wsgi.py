import sys
import os
from dotenv import load_dotenv

# === Detect whether running on EC2 or Local ===
if os.name == "posix" and os.uname().nodename.startswith("ip-"):
    # EC2 (Linux)
    project_home = '/var/www/71-DeployingYourBlog-start'
else:
    # Local (Windows or Mac)
    project_home = os.path.abspath(os.path.dirname(__file__))


if project_home not in sys.path:
    sys.path.insert(0, project_home)

# === Load .env if running locally (optional) ===
load_dotenv(os.path.join(project_home, '.env'))
load_dotenv(os.path.join(project_home, '.env.local'))  # <- add this line
print("Loaded DB_URI:", os.getenv("DB_URI"))


# === Ensure Flask Key is loaded (required for CSRF) ===
if not os.environ.get("FLASK_KEY"):
    os.environ["FLASK_KEY"] = os.environ.get("FLASK_KEY", "fallback-secret-key")

# === Launch Flask App ===
from main import app as application

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000)



