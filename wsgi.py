import os
import sys
from dotenv import load_dotenv

# === Detect EC2 vs. Local Environment ===
if os.name == "posix" and os.uname().nodename.startswith("ip-"):
    # Running on EC2
    project_home = '/var/www/71-DeployingYourBlog-start'
else:
    # Running locally
    project_home = os.path.abspath(os.path.dirname(__file__))

# Add to sys.path if not already
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# === Load .env (supports both .env and .env.local) ===
for fname in ['.env', '.env.local']:
    dotenv_path = os.path.join(project_home, fname)
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
        if __name__ == "__main__":
            print(f"Loaded {fname} from {dotenv_path}")

# === Fallback for critical env variables (e.g., FLASK_KEY) ===
if not os.environ.get("FLASK_KEY"):
    os.environ["FLASK_KEY"] = "fallback-secret-key"
    if __name__ == "__main__":
        print("FLASK_KEY not found. Using fallback.")

# === Import and Launch Flask App ===
from main import app as application  # For WSGI/Gunicorn

if __name__ == "__main__":
    print("🔧 Running in standalone mode (development server)")
    print("DB_URI =", os.getenv("DB_URI"))
    application.run(host="0.0.0.0", port=5000)




