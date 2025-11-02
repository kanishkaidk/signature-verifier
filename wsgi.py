"""
WSGI entry point for deployment (Render, Heroku, etc.)
"""
import sys
import os

# Add the project root to Python path
# This ensures 'backend' module can be imported
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the Flask app
from backend.app import app

# For local testing
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

