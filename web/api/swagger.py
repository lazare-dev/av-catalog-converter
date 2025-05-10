"""
Swagger UI integration for API documentation
"""
import os
from pathlib import Path
from flask import Blueprint, send_from_directory, render_template_string

# Create blueprint
swagger_bp = Blueprint('swagger', __name__, url_prefix='/api/docs')

# HTML template for Swagger UI
SWAGGER_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AV Catalog Converter API Documentation</title>
    <link rel="stylesheet" type="text/css" href="https://unpkg.com/swagger-ui-dist@4.5.0/swagger-ui.css" />
    <link rel="icon" type="image/png" href="https://unpkg.com/swagger-ui-dist@4.5.0/favicon-32x32.png" sizes="32x32" />
    <link rel="icon" type="image/png" href="https://unpkg.com/swagger-ui-dist@4.5.0/favicon-16x16.png" sizes="16x16" />
    <style>
        html {
            box-sizing: border-box;
            overflow: -moz-scrollbars-vertical;
            overflow-y: scroll;
        }
        
        *,
        *:before,
        *:after {
            box-sizing: inherit;
        }
        
        body {
            margin: 0;
            background: #fafafa;
        }
    </style>
</head>
<body>
    <div id="swagger-ui"></div>
    
    <script src="https://unpkg.com/swagger-ui-dist@4.5.0/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@4.5.0/swagger-ui-standalone-preset.js"></script>
    <script>
        window.onload = function() {
            const ui = SwaggerUIBundle({
                url: "{{ spec_url }}",
                dom_id: '#swagger-ui',
                deepLinking: true,
                presets: [
                    SwaggerUIBundle.presets.apis,
                    SwaggerUIStandalonePreset
                ],
                plugins: [
                    SwaggerUIBundle.plugins.DownloadUrl
                ],
                layout: "StandaloneLayout"
            });
            
            window.ui = ui;
        };
    </script>
</body>
</html>
"""

@swagger_bp.route('/')
def swagger_ui():
    """Serve Swagger UI"""
    return render_template_string(SWAGGER_TEMPLATE, spec_url='/api/docs/openapi.yaml')

@swagger_bp.route('/openapi.yaml')
def swagger_spec():
    """Serve OpenAPI specification"""
    docs_dir = Path(__file__).parent.parent.parent / 'docs'
    return send_from_directory(docs_dir, 'openapi.yaml')

def register_swagger(app):
    """Register Swagger blueprint with Flask app"""
    app.register_blueprint(swagger_bp)
