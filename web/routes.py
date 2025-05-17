# web/routes.py
"""
API route definitions for AV Catalog Standardizer
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import json
import pandas as pd

from flask import Flask, request, jsonify, send_from_directory, Blueprint
from werkzeug.utils import secure_filename

from core.file_parser.parser_factory import ParserFactory
from services.structure.structure_analyzer import StructureAnalyzer
from services.mapping.field_mapper import FieldMapper
from services.category.category_extractor import CategoryExtractor
from services.normalization.value_normalizer import ValueNormalizer
from utils.helpers.validation_helpers import validate_output
from utils.logging.progress_logger import ProgressLogger

# Create routes blueprint
api_bp = Blueprint('api', __name__)

# Configure logging
logger = logging.getLogger(__name__)

# Create shared progress logger
progress_logger = ProgressLogger()

# Temporary uploads directory
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), 'av_catalog_uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Active processing jobs
active_jobs = {}

@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint

    Returns:
        JSON response with health status
    """
    from config.settings import APP_CONFIG
    import time

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
        app_info['llm_info'] = llm_stats
    except Exception as e:
        logger.warning(f"Could not get LLM information: {str(e)}")

    return jsonify(app_info)

@api_bp.route('/upload', methods=['POST'])
def upload_file():
    """
    Handle file upload

    Returns:
        JSON response with upload result
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Secure filename and save
    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    logger.info(f"File uploaded: {filename}")

    # Try to parse the file to verify it's valid
    try:
        parser = ParserFactory.create_parser(file_path)
        sample = parser.get_sample(5)

        # Generate job ID
        import uuid
        job_id = str(uuid.uuid4())

        # Try to get row count
        row_count = 0
        if hasattr(parser, 'get_row_count'):
            try:
                row_count = parser.get_row_count()
            except:
                pass

        # If we couldn't get row count, use sample length
        if row_count == 0 and not sample.empty:
            row_count = len(sample)

        # Store job info
        active_jobs[job_id] = {
            'file_path': file_path,
            'filename': filename,
            'status': 'uploaded',
            'progress': 0,
            'row_count': row_count,
            'sample_data': sample.to_dict(orient='records') if not sample.empty else {}
        }

        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'job_id': job_id,
            'filename': filename,
            'row_count': row_count,
            'sample_data': sample.to_json(orient='records')
        })

    except Exception as e:
        logger.error(f"Error parsing uploaded file: {str(e)}")
        return jsonify({'error': f'Invalid file format: {str(e)}'}), 400

@api_bp.route('/analyze', methods=['POST'])
def analyze_file():
    """
    Analyze file structure

    Returns:
        JSON response with analysis result
    """
    # Initialize job tracking
    import uuid
    from datetime import datetime

    # Check if job_id is provided in the request
    existing_job_id = request.form.get('job_id')

    if existing_job_id and existing_job_id in active_jobs:
        # Use existing job
        job_id = existing_job_id
        job = active_jobs[job_id]
        file_path = job.get('file_path')

        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'File not found for the provided job ID'}), 404

        logger.info(f"Using existing file for job {job_id}")
    else:
        # Create new job
        job_id = str(uuid.uuid4())
        job = {
            'id': job_id,
            'status': 'pending',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'error': None
        }

        # Store job in active_jobs dictionary
        active_jobs[job_id] = job

        # Get file from request
        if 'file' not in request.files:
            job['status'] = 'error'
            job['error'] = 'No file provided'
            return jsonify({'error': 'No file provided or job ID not found'}), 400

        file = request.files['file']
        if file.filename == '':
            job['status'] = 'error'
            job['error'] = 'No file selected'
            return jsonify({'error': 'No file selected'}), 400

        # Save file temporarily
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_DIR, filename)
            file.save(file_path)
            job['file_path'] = file_path
            job['filename'] = filename
        except Exception as e:
            job['status'] = 'error'
            job['error'] = f'Error saving file: {str(e)}'
            return jsonify({'error': f'Error saving file: {str(e)}'}), 500

    try:
        # Update job status
        job['status'] = 'analyzing'

        # Parse file
        parser = ParserFactory.create_parser(file_path)
        data = parser.parse()

        # Analyze structure
        progress_logger.start_task("Analyzing file structure")
        analyzer = StructureAnalyzer()
        structure_info = analyzer.analyze(data)
        progress_logger.complete_task("Structure analysis complete")

        # Update job with structure info
        job['structure_info'] = structure_info
        job['status'] = 'analyzed'
        job['progress'] = 30

        # Extract potential field mappings
        field_mappings = structure_info.get('possible_field_mappings', {})

        # Update job with row count if available
        if 'row_count' not in job and 'row_count' in structure_info:
            job['row_count'] = structure_info['row_count']

        # If we have effective rows, use that as the primary row count
        if 'effective_rows' in structure_info:
            job['row_count'] = structure_info['effective_rows']

        return jsonify({
            'success': True,
            'message': 'File analyzed successfully',
            'job_id': job_id,  # Include job_id in the response
            'field_mappings': field_mappings,
            'column_info': structure_info.get('column_analysis', {}),
            'data_quality': structure_info.get('data_quality_issues', []),
            'structure': {
                'row_count': job.get('row_count', 0),
                'column_count': structure_info.get('column_count', 0),
                'effective_rows': structure_info.get('effective_rows', 0),
                'data_rows': structure_info.get('data_rows', 0)
            }
        })

    except Exception as e:
        logger.error(f"Error analyzing file: {str(e)}")
        if job_id in active_jobs:
            active_jobs[job_id]['status'] = 'error'
            active_jobs[job_id]['error'] = str(e)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@api_bp.route('/map/<job_id>', methods=['POST'])
def map_fields(job_id):
    """
    Map fields to standard schema

    Args:
        job_id (str): Job ID

    Returns:
        JSON response with mapping result
    """
    if job_id not in active_jobs:
        return jsonify({'error': 'Invalid job ID'}), 404

    job = active_jobs[job_id]
    file_path = job['file_path']

    # Get user-specified mappings from request
    user_mappings = request.json.get('mappings', {})

    try:
        # Update job status
        job['status'] = 'mapping'

        # Parse file
        parser = ParserFactory.create_parser(file_path)
        data = parser.parse()

        # Get structure info from job or analyze if not available
        structure_info = job.get('structure_info', {})
        if not structure_info:
            analyzer = StructureAnalyzer()
            structure_info = analyzer.analyze(data)
            job['structure_info'] = structure_info

        # Map fields
        progress_logger.start_task("Mapping fields")
        field_mapper = FieldMapper()

        # Inject user mappings if provided
        if user_mappings:
            field_mapper.user_mappings = user_mappings

        mapped_data = field_mapper.map(data, structure_info)
        progress_logger.complete_task("Field mapping complete")

        # Update job status
        job['status'] = 'mapped'
        job['progress'] = 60
        job['mapping_results'] = field_mapper.last_mapping_results

        # Generate sample of mapped data
        sample = mapped_data.head(10).to_dict(orient='records')

        # Get LLM stats if available
        llm_stats = {}
        if hasattr(field_mapper.llm_client, 'get_stats'):
            llm_stats = field_mapper.llm_client.get_stats()

        return jsonify({
            'success': True,
            'message': 'Fields mapped successfully',
            'mapping_results': field_mapper.last_mapping_results,
            'sample_data': sample,
            'missing_required': field_mapper.last_mapping_results.get('missing_required', []),
            'llm_stats': llm_stats
        })

    except Exception as e:
        logger.error(f"Error mapping fields: {str(e)}")
        job['status'] = 'error'
        job['error'] = str(e)
        return jsonify({'error': f'Mapping failed: {str(e)}'}), 500

@api_bp.route('/process/<job_id>', methods=['POST'])
def process_file(job_id):
    """
    Process the file through the entire pipeline

    Args:
        job_id (str): Job ID

    Returns:
        JSON response with processing result
    """
    if job_id not in active_jobs:
        return jsonify({'error': 'Invalid job ID'}), 404

    job = active_jobs[job_id]
    file_path = job['file_path']

    # Get configuration from request
    config = request.json or {}
    user_mappings = config.get('mappings', {})
    output_format = config.get('output_format', 'csv')

    try:
        # Update job status
        job['status'] = 'processing'
        job['progress'] = 0

        # Parse file
        progress_logger.start_task("Parsing input file")
        parser = ParserFactory.create_parser(file_path)
        data = parser.parse()
        progress_logger.complete_task("File parsed successfully")
        job['progress'] = 20

        # Analyze structure
        progress_logger.start_task("Analyzing file structure")
        analyzer = StructureAnalyzer()
        structure_info = analyzer.analyze(data)
        progress_logger.complete_task("Structure analysis complete")
        job['progress'] = 40

        # Map fields
        progress_logger.start_task("Mapping fields")
        field_mapper = FieldMapper()
        if user_mappings:
            field_mapper.user_mappings = user_mappings
        mapped_data = field_mapper.map(data, structure_info)
        progress_logger.complete_task("Field mapping complete")
        job['progress'] = 60

        # Store LLM stats if available
        if hasattr(field_mapper.llm_client, 'get_stats'):
            job['llm_stats'] = field_mapper.llm_client.get_stats()

        # Extract categories
        progress_logger.start_task("Extracting categories")
        category_extractor = CategoryExtractor()
        categorized_data = category_extractor.extract_categories(mapped_data)
        progress_logger.complete_task("Category extraction complete")
        job['progress'] = 80

        # Normalize values
        progress_logger.start_task("Normalizing values")
        normalizer = ValueNormalizer()
        normalized_data = normalizer.normalize(categorized_data)
        progress_logger.complete_task("Value normalization complete")

        # Validate output
        progress_logger.start_task("Validating output")
        validated_data = validate_output(normalized_data)
        progress_logger.complete_task("Validation complete")
        job['progress'] = 90

        # Generate output file
        output_filename = f"{os.path.splitext(job['filename'])[0]}_standardized"
        output_path = os.path.join(UPLOAD_DIR, output_filename)

        if output_format == 'csv':
            output_file = f"{output_path}.csv"
            validated_data.to_csv(output_file, index=False)
        else:  # excel
            output_file = f"{output_path}.xlsx"
            validated_data.to_excel(output_file, index=False)

        # Update job status
        job['status'] = 'completed'
        job['progress'] = 100
        job['output_file'] = output_file
        job['output_format'] = output_format

        return jsonify({
            'success': True,
            'message': 'File processed successfully',
            'output_file': os.path.basename(output_file),
            'job_status': job['status'],
            'llm_stats': job.get('llm_stats', {})
        })

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        job['status'] = 'error'
        job['error'] = str(e)
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@api_bp.route('/download/<job_id>', methods=['GET'])
def download_file(job_id):
    """
    Download processed file

    Args:
        job_id (str): Job ID

    Returns:
        Processed file for download
    """
    if job_id not in active_jobs:
        return jsonify({'error': 'Invalid job ID'}), 404

    job = active_jobs[job_id]

    if job['status'] != 'completed' or 'output_file' not in job:
        return jsonify({'error': 'Output file not available'}), 400

    output_file = job['output_file']
    filename = os.path.basename(output_file)

    return send_from_directory(
        os.path.dirname(output_file),
        filename,
        as_attachment=True
    )

@api_bp.route('/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """
    Get job status

    Args:
        job_id (str): Job ID

    Returns:
        JSON response with job status
    """
    if job_id not in active_jobs:
        return jsonify({'error': 'Invalid job ID'}), 404

    job = active_jobs[job_id]

    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'progress': job['progress'],
        'filename': job['filename'],
        'row_count': job.get('row_count', 0),
        'error': job.get('error'),
        'llm_stats': job.get('llm_stats', {})
    })

@api_bp.route('/preview/<job_id>', methods=['GET'])
def preview_data(job_id):
    """
    Preview processed data

    Args:
        job_id (str): Job ID

    Returns:
        JSON response with data preview
    """
    if job_id not in active_jobs:
        return jsonify({'error': 'Invalid job ID'}), 404

    job = active_jobs[job_id]

    if job['status'] not in ['mapped', 'completed']:
        return jsonify({'error': 'Processed data not available'}), 400

    try:
        # If completed, load from output file
        if job['status'] == 'completed' and 'output_file' in job:
            output_file = job['output_file']

            if output_file.endswith('.csv'):
                data = pd.read_csv(output_file)
            else:  # excel
                data = pd.read_excel(output_file)

            preview = data.head(50).to_dict(orient='records')
            columns = list(data.columns)

        else:
            # Otherwise, use saved sample data
            preview = job.get('sample_data', {})
            columns = list(preview[0].keys()) if preview else []

        return jsonify({
            'success': True,
            'preview': preview,
            'columns': columns
        })

    except Exception as e:
        logger.error(f"Error generating preview: {str(e)}")
        return jsonify({'error': f'Preview failed: {str(e)}'}), 500

@api_bp.route('/cleanup/<job_id>', methods=['DELETE'])
def cleanup_job(job_id):
    """
    Clean up job resources

    Args:
        job_id (str): Job ID

    Returns:
        JSON response with cleanup result
    """
    if job_id not in active_jobs:
        return jsonify({'error': 'Invalid job ID'}), 404

    job = active_jobs[job_id]

    try:
        # Remove input file
        if 'file_path' in job and os.path.exists(job['file_path']):
            os.remove(job['file_path'])

        # Remove output file
        if 'output_file' in job and os.path.exists(job['output_file']):
            os.remove(job['output_file'])

        # Remove job from active jobs
        del active_jobs[job_id]

        return jsonify({
            'success': True,
            'message': 'Job cleaned up successfully'
        })

    except Exception as e:
        logger.error(f"Error cleaning up job: {str(e)}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500
