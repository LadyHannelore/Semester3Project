# API initialization and registration
from flask import Blueprint

# Import all API route blueprints
from .classroom_routes import api as classroom_api

# Create function to register all blueprints
def register_api_routes(app):
    """
    Register all API route blueprints with the Flask app
    
    Args:
        app: Flask application instance
    """
    # Create main API blueprint
    api_bp = Blueprint('api', __name__, url_prefix='/api')
    
    # Register route blueprints with the main API blueprint
    api_bp.register_blueprint(classroom_api, url_prefix='/classrooms')
    
    # Register the main blueprint with the app
    app.register_blueprint(api_bp)
