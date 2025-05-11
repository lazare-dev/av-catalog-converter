"""
API routes for AV Catalog Standardizer
"""
from flask import Blueprint

# Create API blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# Import routes to register them with the blueprint
from web.routes import *

# Import controllers
from web.api.mapping_controller import MappingController
from web.api.upload_controller import UploadController
from web.api.preview_controller import PreviewController

# Initialize controllers
mapping_controller = MappingController()
upload_controller = UploadController()
preview_controller = PreviewController()
