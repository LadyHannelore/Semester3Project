"""
Error handlers and custom exceptions for ClassForge backend.
This module provides custom exceptions and error handling utilities.
"""

from flask import jsonify


class ClassForgeException(Exception):
    """Base exception for ClassForge application"""
    status_code = 500
    
    def __init__(self, message, status_code=None, payload=None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
    
    def to_dict(self):
        """Convert exception to dictionary for JSON response"""
        result = dict(self.payload or ())
        result['error'] = self.message
        result['success'] = False
        return result


class ValidationError(ClassForgeException):
    """Exception raised for validation errors"""
    status_code = 400


class ResourceNotFoundError(ClassForgeException):
    """Exception raised when a requested resource is not found"""
    status_code = 404


class ConfigurationError(ClassForgeException):
    """Exception raised for configuration errors"""
    status_code = 500


def register_error_handlers(app):
    """
    Register error handlers with the Flask application
    
    Args:
        app: Flask application instance
    """
    @app.errorhandler(ClassForgeException)
    def handle_classforge_exception(error):
        """Handle ClassForge custom exceptions"""
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response
    
    @app.errorhandler(404)
    def handle_not_found(error):
        """Handle 404 errors"""
        return jsonify({
            'success': False,
            'error': 'Resource not found',
            'message': str(error)
        }), 404
    
    @app.errorhandler(400)
    def handle_bad_request(error):
        """Handle 400 errors"""
        return jsonify({
            'success': False,
            'error': 'Bad request',
            'message': str(error)
        }), 400
    
    @app.errorhandler(500)
    def handle_internal_server_error(error):
        """Handle 500 errors"""
        app.logger.exception("Internal server error: %s", str(error))
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'An unexpected error occurred'
        }), 500
    
    @app.errorhandler(Exception)
    def handle_generic_exception(error):
        """Handle all other exceptions"""
        app.logger.exception("Unhandled exception: %s", str(error))
        return jsonify({
            'success': False,
            'error': 'Server error',
            'message': str(error) if app.config.get('DEBUG', False) else 'An unexpected error occurred'
        }), 500
