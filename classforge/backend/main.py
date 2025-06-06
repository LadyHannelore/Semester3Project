# Main entry point for ClassForge backend
import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from backend.api import register_api_routes
from backend.utils.logging_config import setup_logging
from backend.utils.error_handlers import register_error_handlers

# Application version
VERSION = '1.0.0'

# Import database modules
from backend.models.database import db
from backend.models.migrations import register_db_cli

def create_app(test_config=None):
    """Create and configure the Flask application"""
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # Default configuration
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        VERSION=VERSION,
        DEBUG=os.environ.get('FLASK_ENV', 'production') == 'development',
        SQLALCHEMY_DATABASE_URI=os.environ.get('DATABASE_URL', 'sqlite:///instance/classforge.db'),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    
    # Apply test config if provided
    if test_config is not None:
        app.config.from_mapping(test_config)
    
    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    # Initialize database
    db.init_app(app)
    
    # Register database CLI commands
    register_db_cli(app)
    
    # Set up CORS
    CORS(app)
    
    # Set up logging
    setup_logging(app)
    
    # Register API routes
    register_api_routes(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Root route
    @app.route('/')
    def index():
        app.logger.info("Root endpoint accessed")
        return jsonify({
            'service': 'ClassForge API',
            'version': app.config['VERSION'],
            'status': 'running',
            'documentation': '/api/docs',
            'environment': os.environ.get('FLASK_ENV', 'production')
        })
    
    # Health check endpoint
    @app.route('/health')
    def health():
        app.logger.debug("Health check endpoint accessed")
        return jsonify({'status': 'healthy'})
    
    app.logger.info(f"ClassForge API initialized (Version: {VERSION})")
    return app

if __name__ == '__main__':
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5001))
    
    # Create the app
    app = create_app()
    
    # Run the app
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_ENV', 'production') == 'development')
