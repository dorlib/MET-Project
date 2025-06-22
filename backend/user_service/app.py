#!/usr/bin/env python3
# User Service for MET Brain Metastasis Segmentation Service

from flask import Flask
from flask_cors import CORS
import logging
import time
from utils.database import init_db
from controllers.controllers import auth_bp, user_bp, scan_bp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Initialize database (with retry mechanism)
retry_count = 5
for i in range(retry_count):
    try:
        logger.info("Attempting to initialize database tables...")
        init_db()
        logger.info("Database tables initialized successfully")
        break
    except Exception as e:
        if i < retry_count - 1:
            wait_time = (i + 1) * 2
            logger.warning(f"Database initialization failed, retrying in {wait_time} seconds. Error: {str(e)}")
            time.sleep(wait_time)
        else:
            logger.error(f"Failed to initialize database after {retry_count} attempts: {str(e)}")
            raise

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='')
app.register_blueprint(user_bp, url_prefix='')
app.register_blueprint(scan_bp, url_prefix='')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
