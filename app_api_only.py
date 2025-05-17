"""
AV Catalog Standardizer - API-only version
Automates conversion of diverse audio-visual equipment catalogs into a standardized format

This version removes all frontend components and focuses only on the API functionality.
"""
import os
import argparse
import logging
import traceback
import time
import json
from pathlib import Path
from flask import Flask, request, jsonify, send_file, redirect
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import tempfile

from config.settings import APP_CONFIG
from config.logging_config import setup_logging
from core.file_parser.parser_factory import ParserFactory
from services.structure.structure_analyzer import StructureAnalyzer
from services.mapping.field_mapper import FieldMapper
from services.category.category_extractor import CategoryExtractor
from services.normalization.value_normalizer import ValueNormalizer
from utils.helpers.validation_helpers import validate_output
from utils.logging.progress_logger import ProgressLogger
from utils.logging.logger import Logger
from web.api.swagger import register_swagger
from web.api.logging_controller import logging_api
from web.api.validation_controller import ValidationController
from web.cors import add_cors_headers

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size
# Use a dedicated uploads directory that is mounted as a volume in Docker
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize analysis results storage
app.analysis_results = {}

# Create a directory for persistent job storage
JOB_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job_storage')
os.makedirs(JOB_STORAGE_DIR, exist_ok=True)

def load_existing_jobs():
    """Load existing job data from disk at startup"""
    try:
        # Get all JSON files in the job storage directory
        job_files = [f for f in os.listdir(JOB_STORAGE_DIR) if f.endswith('.json')]
        logger.info(f"Found {len(job_files)} job files on disk")

        # Load each job file
        for job_file in job_files:
            try:
                job_id = job_file.replace('.json', '')
                job_data = load_job_data(job_id)
                if job_data:
                    app.analysis_results[job_id] = job_data
                    logger.info(f"Loaded job data for job ID: {job_id}")
            except Exception as e:
                logger.error(f"Error loading job file {job_file}: {str(e)}")

        logger.info(f"Loaded {len(app.analysis_results)} jobs from disk")
    except Exception as e:
        logger.error(f"Error loading existing jobs: {str(e)}")
        logger.debug(traceback.format_exc())

def save_job_data(job_id, data):
    """Save job data to disk for persistence"""
    try:
        # Create a copy of the data without non-serializable objects
        serializable_data = {
            'job_id': job_id,
            'timestamp': time.time(),
            'file_path': data.get('file_path'),
            'filename': data.get('filename')
        }

        # Ensure the job storage directory exists
        os.makedirs(JOB_STORAGE_DIR, exist_ok=True)

        # Save to disk
        job_file = os.path.join(JOB_STORAGE_DIR, f"{job_id}.json")

        # Create a backup of the existing file if it exists
        if os.path.exists(job_file):
            backup_file = os.path.join(JOB_STORAGE_DIR, f"{job_id}_backup.json")
            try:
                import shutil
                shutil.copy2(job_file, backup_file)
                logger.info(f"Created backup of job file: {backup_file}")
            except Exception as backup_error:
                logger.warning(f"Failed to create backup of job file: {str(backup_error)}")

        # Save the new job data
        with open(job_file, 'w') as f:
            json.dump(serializable_data, f)

        logger.info(f"Saved job data to disk: {job_file}")

        # Also update the in-memory storage
        app.analysis_results[job_id] = data
        logger.info(f"Updated in-memory job data for job ID: {job_id}")

        return True
    except Exception as e:
        logger.error(f"Error saving job data: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def load_job_data(job_id):
    """Load job data from disk"""
    try:
        # First try the standard job file
        job_file = os.path.join(JOB_STORAGE_DIR, f"{job_id}.json")

        # If the standard file doesn't exist, try the backup
        if not os.path.exists(job_file):
            logger.warning(f"Job file not found: {job_file}")
            backup_file = os.path.join(JOB_STORAGE_DIR, f"{job_id}_backup.json")

            if os.path.exists(backup_file):
                logger.info(f"Found backup job file: {backup_file}")
                job_file = backup_file
            else:
                logger.warning(f"Backup job file not found: {backup_file}")
                return None

        # Load the job data from the file
        with open(job_file, 'r') as f:
            data = json.load(f)

        logger.info(f"Loaded job data from disk: {job_file}")
        return data
    except Exception as e:
        logger.error(f"Error loading job data: {str(e)}")
        logger.debug(traceback.format_exc())
        return None

# Register blueprints
app.register_blueprint(logging_api, url_prefix='/api')

# Setup logging with verbose DEBUG level
setup_logging(logging.DEBUG)
logger = Logger.get_logger(__name__)
logger.info("Application starting with verbose logging enabled")

# Log the job storage directory
logger.info(f"Using job storage directory: {JOB_STORAGE_DIR}")

# Load existing jobs from disk
load_existing_jobs()

# Initialize LLM components
try:
    from core.llm.llm_factory import LLMFactory
    # Pre-initialize the LLM client to ensure it's ready for API requests
    LLMFactory.create_client()
    logger.info("LLM client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing LLM client: {str(e)}")
    logger.debug(traceback.format_exc())

# Add CORS support
add_cors_headers(app)

# Register Swagger UI
register_swagger(app)

# Redirect root to API docs
@app.route('/')
def index():
    """Redirect to API documentation"""
    return redirect('/api/docs/')

# API Routes
@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Health check endpoint

    This endpoint returns basic information about the application status.
    It can be used to verify that the API is running properly.

    Returns:
        JSON with status information
    """
    try:
        # Get application information
        app_info = {
            'status': 'ok',
            'version': APP_CONFIG['version'],
            'app_name': APP_CONFIG['app_name'],
            'timestamp': time.time(),
            'environment': os.environ.get('FLASK_ENV', 'production')
        }

        # Add LLM information if available
        try:
            from core.llm.llm_factory import LLMFactory
            llm_stats = LLMFactory.get_stats()
            app_info['llm'] = llm_stats
        except Exception as e:
            logger.warning(f"Could not get LLM information: {str(e)}")

        # Log health check
        logger.info("Health check request")
        return jsonify(app_info)

    except Exception as e:
        # Log error
        logger.error(f"Health check failed: {str(e)}", exc_info=True)

        # Return error response
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': time.time()
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """
    Upload catalog file

    This endpoint accepts a file upload and saves it for later processing.
    It returns a job ID that can be used to reference the uploaded file.

    Returns:
        JSON with job ID and file info
    """
    temp_file_path = None

    try:
        # Log request details for debugging
        logger.info(f"Upload request received: {request.method} {request.path}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request files: {list(request.files.keys())}")
        logger.info(f"Request form data: {dict(request.form)}")

        # Check if file was uploaded
        if 'file' not in request.files:
            logger.warning("Upload attempt with no file provided")
            return jsonify({
                'success': False,
                'error': 'No file provided',
                'details': 'Please include a file in your request'
            }), 400

        file = request.files['file']
        logger.info(f"File object: {file}, filename: {file.filename}")

        # Check if filename is empty
        if file.filename == '':
            logger.warning("Upload attempt with empty filename")
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'details': 'The uploaded file has no name'
            }), 400

        # Generate a unique filename to avoid conflicts
        import uuid
        original_filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        filename = f"{unique_id}_{original_filename}"
        logger.info(f"Generated unique filename: {filename}")

        # Ensure uploads directory exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        logger.info(f"Uploads directory: {app.config['UPLOAD_FOLDER']}")

        # Save file to the uploads directory
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to: {temp_file_path}")
        file.save(temp_file_path)

        # Verify file was saved
        if os.path.exists(temp_file_path):
            file_size = os.path.getsize(temp_file_path)
            logger.info(f"File saved successfully: {temp_file_path}, size: {file_size} bytes")
        else:
            logger.error(f"File was not saved: {temp_file_path}")
            return jsonify({
                'success': False,
                'error': 'File save failed',
                'details': 'The file could not be saved to the server'
            }), 500

        # Generate a job ID
        job_id = str(uuid.uuid4())
        logger.info(f"Generated job ID: {job_id}")

        # Save job data
        job_data = {
            'job_id': job_id,
            'filename': original_filename,
            'file_path': temp_file_path,
            'upload_time': time.time(),
            'status': 'uploaded',
            'company': request.form.get('company', 'unknown')
        }

        logger.info(f"Job data: {job_data}")

        # Save to disk and memory
        save_result = save_job_data(job_id, job_data)

        if not save_result:
            logger.error(f"Failed to save job data for job ID: {job_id}")
            return jsonify({
                'success': False,
                'error': 'Job data save failed',
                'details': 'The job data could not be saved to the server'
            }), 500

        # Verify job data was saved
        if job_id in app.analysis_results:
            logger.info(f"Job data saved in memory: {job_id}")
        else:
            logger.warning(f"Job data not found in memory after save: {job_id}")

        # Return job ID and file info
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'job_id': job_id,
            'filename': original_filename
        })

    except RequestEntityTooLarge as e:
        logger.error(f"File too large: {str(e)}")
        return jsonify({
            'error': 'File too large',
            'details': 'The uploaded file exceeds the maximum allowed size of 100MB',
            'max_size_mb': 100
        }), 413
    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Server error',
            'details': str(e)
        }), 500

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AV Catalog Standardizer')
    parser.add_argument('--api', action='store_true', help='Run as API server')
    parser.add_argument('--port', type=int, default=8080, help='Port for API server')
    return parser.parse_args()

@app.route('/api/analyze', methods=['POST'])
def analyze_file():
    """
    Analyze catalog file structure

    This endpoint accepts either:
    1. A file upload (multipart/form-data with 'file' field)
    2. A job ID from a previous upload (either as form data or JSON)

    It analyzes the structure of the file and returns information about it.

    Returns:
        JSON with file structure analysis or an error response
    """
    try:
        # Log request details for debugging
        logger.info(f"Analyze request received: {request.method} {request.path}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content type: {request.content_type}")

        # Variable to store the job ID and file path
        job_id = None
        file_path = None
        temp_file_path = None

        # Check if this is a multipart/form-data request (file upload or form-submitted job_id)
        if request.content_type and 'multipart/form-data' in request.content_type:
            logger.info("Processing multipart/form-data request")
            logger.info(f"Request files: {list(request.files.keys())}")
            logger.info(f"Request form data: {dict(request.form)}")

            # Check if job_id is provided in form data
            if 'job_id' in request.form and request.form['job_id']:
                job_id = request.form['job_id']
                logger.info(f"Using job ID from form data: {job_id}")

            # Check if file is uploaded
            elif 'file' in request.files and request.files['file'].filename != '':
                # Handle file upload
                file = request.files['file']
                logger.info(f"File uploaded: {file.filename}")

                # Generate a unique filename to avoid conflicts
                import uuid
                original_filename = secure_filename(file.filename)
                unique_id = str(uuid.uuid4())
                filename = f"{unique_id}_{original_filename}"

                # Save file to the uploads directory
                temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                logger.info(f"Saving file to: {temp_file_path}")
                file.save(temp_file_path)

                # Generate a new job ID for this upload
                job_id = str(uuid.uuid4())

                # Save job data
                job_data = {
                    'job_id': job_id,
                    'filename': original_filename,
                    'file_path': temp_file_path,
                    'upload_time': time.time(),
                    'status': 'uploaded'
                }

                # Save to disk and memory
                save_result = save_job_data(job_id, job_data)

                if not save_result:
                    logger.error(f"Failed to save job data for job ID: {job_id}")
                    return jsonify({
                        'success': False,
                        'error': 'Job data save failed',
                        'details': 'The job data could not be saved to the server'
                    }), 500

                file_path = temp_file_path
            else:
                logger.warning("No file or job_id provided in multipart/form-data request")
                return jsonify({
                    'success': False,
                    'error': 'Missing data',
                    'details': 'Please provide either a file or a job_id'
                }), 400

        # Check if this is a JSON request
        elif request.is_json:
            logger.info("Processing JSON request")
            data = request.json
            logger.info(f"Request JSON data: {data}")

            # Get job ID from JSON
            job_id = data.get('job_id')
            if not job_id:
                logger.warning("No job ID provided in JSON request")
                return jsonify({
                    'success': False,
                    'error': 'Missing job ID',
                    'details': 'Please provide a job_id in your JSON request'
                }), 400

        # If we couldn't determine the request type
        else:
            logger.warning(f"Unsupported content type: {request.content_type}")
            return jsonify({
                'success': False,
                'error': 'Invalid content type',
                'details': 'Request must be either multipart/form-data or application/json'
            }), 400

        # If we have a job ID but no file path yet, load the job data
        if job_id and not file_path:
            logger.info(f"Loading job data for job ID: {job_id}")

            # First check in-memory storage
            if job_id in app.analysis_results:
                job_data = app.analysis_results[job_id]
                logger.info(f"Found job data in memory: {job_id}")
            else:
                # Try to load from disk
                job_data = load_job_data(job_id)

            if not job_data:
                logger.warning(f"No job data found for job ID: {job_id}")
                return jsonify({
                    'success': False,
                    'error': 'Invalid job ID',
                    'details': f'No job data found for job ID: {job_id}'
                }), 404

            # Get file path from job data
            file_path = job_data.get('file_path')
            if not file_path or not os.path.exists(file_path):
                logger.warning(f"File not found for job ID: {job_id}")
                return jsonify({
                    'success': False,
                    'error': 'File not found',
                    'details': f'The file for job ID {job_id} was not found'
                }), 404

        # At this point, we should have both a job_id and a file_path
        if not job_id or not file_path:
            logger.error("Failed to determine job ID or file path")
            return jsonify({
                'success': False,
                'error': 'Processing error',
                'details': 'Could not determine job ID or file path'
            }), 500

        # Perform file analysis
        try:
            # Parse file
            parser = ParserFactory.create_parser(file_path)
            data = parser.parse()

            # Get basic file info
            columns = list(data.columns)
            row_count = len(data)

            # Get sample data (up to 5 rows)
            sample_data = data.head(5).to_dict(orient='records') if not data.empty else []

            # Return analysis results
            return jsonify({
                'success': True,
                'message': 'File analysis completed',
                'job_id': job_id,
                'filename': os.path.basename(file_path),
                'structure': {
                    'status': 'analyzed',
                    'row_count': row_count,
                    'column_count': len(columns),
                    'columns': columns
                },
                'sample_data': sample_data
            })

        except Exception as analysis_error:
            logger.error(f"Error analyzing file content: {str(analysis_error)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': 'Analysis failed',
                'details': f'Error analyzing file content: {str(analysis_error)}'
            }), 500

    except Exception as e:
        logger.error(f"Error in analyze_file endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Analysis failed',
            'details': str(e)
        }), 500

def main():
    """Main application entry point"""
    # Parse arguments
    args = parse_arguments()

    # Run as API server
    logger.info(f"Starting API server on port {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)
    return 0

if __name__ == '__main__':
    main()
