"""
Logging configuration for ClassForge backend.
This module sets up proper logging for the application.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime

def setup_logging(app=None):
    """
    Configure logging for the application

    Args:
        app (Flask, optional): Flask application instance. Defaults to None.
    """
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up file handler with rotation
    log_file = os.path.join(log_dir, f'classforge_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=10)  # 10MB max file size, keep 10 backups
    file_handler.setLevel(logging.INFO)
    
    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # If Flask app provided, set up Flask logging
    if app:
        # Configure Flask app logger
        app.logger.setLevel(logging.INFO)
        for handler in root_logger.handlers:
            app.logger.addHandler(handler)
        
        # Log application startup
        app.logger.info(f"Starting ClassForge application (Version: {app.config.get('VERSION', '1.0.0')})")
        
        # Set up request logging
        @app.before_request
        def log_request_info():
            app.logger.debug('Request: %s %s', request.method, request.path)
        
        # Set up error logging
        @app.errorhandler(Exception)
        def log_exception(error):
            app.logger.exception("Unhandled exception: %s", str(error))
            return jsonify({'error': 'Internal server error', 'message': str(error)}), 500

    return root_logger
