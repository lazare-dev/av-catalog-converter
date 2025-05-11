"""
CORS support for Flask API
"""
from flask import Flask

def add_cors_headers(app: Flask):
    """
    Add CORS headers to all responses
    
    Args:
        app (Flask): Flask application instance
    """
    @app.after_request
    def add_cors(response):
        """Add CORS headers to response"""
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response
