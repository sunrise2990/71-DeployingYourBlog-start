import sys
import os

# Set path and env
sys.path.insert(0, "/var/www/71-DeployingYourBlog-start")
os.environ["FLASK_ENV"] = "production"

# Import Flask app
from main import app as application



