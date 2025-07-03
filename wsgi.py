import sys
import os

# Add your project directory to the sys.path
sys.path.insert(0, "/var/www/71-DeployingYourBlog-start")

# Set environment variables if needed
os.environ["FLASK_ENV"] = "production"

from main import app as application

