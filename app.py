"""
AV Catalog Standardizer - Main application entry point
Automates conversion of diverse audio-visual equipment catalogs into a standardized format
"""
import os
import argparse
import logging
import traceback
import time
from pathlib import Path
import json
from flask import Flask, request, jsonify, send_file, redirect
from werkzeug.utils import secure_filename
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
from web.cors import add_cors_headers

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()

# Setup logging
setup_logging(logging.INFO)
logger = Logger.get_logger(__name__)

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

# Add a redirect from root to API docs
@app.route('/')
def index():
    """Redirect to API documentation"""
    return redirect('/api/docs/')

def process_file(input_path, output_format='csv'):
    """
    Process a catalog file and return standardized data

    Args:
        input_path: Path to the input file
        output_format: Output format (csv, excel, json)

    Returns:
        Tuple of (processed_data, error_message)
    """
    progress = ProgressLogger()
    progress.start_task("Initializing")

    try:
        # Check if file exists
        if not os.path.exists(input_path):
            logger.error(f"File not found: {input_path}")
            progress.fail_task("File not found")
            return None, "File not found"

        # Log file information
        file_size = os.path.getsize(input_path)
        file_extension = os.path.splitext(input_path)[1].lower()
        logger.info(f"Processing file",
                   file_path=input_path,
                   file_size=file_size,
                   file_extension=file_extension,
                   output_format=output_format)

        # Initialize parser based on file extension
        progress.update_task("Parsing input file", 10)
        with logger.context(operation="create_parser", file_path=input_path):
            parser = ParserFactory.create_parser(input_path)
            logger.info(f"Parser created", parser_type=parser.__class__.__name__)
            raw_data = parser.parse()
            logger.info(f"File parsed successfully",
                       rows=len(raw_data) if hasattr(raw_data, '__len__') else 'unknown',
                       columns=len(raw_data.columns) if hasattr(raw_data, 'columns') else 'unknown')

        # Analyze structure of the input data
        progress.update_task("Analyzing data structure", 20)
        with logger.context(operation="analyze_structure"):
            analyzer = StructureAnalyzer()
            structure_info = analyzer.analyze(raw_data)
            logger.info(f"Structure analysis complete",
                       column_count=structure_info.get('column_count', 0))

        # Map fields to standard schema
        progress.update_task("Mapping fields", 40)
        with logger.context(operation="map_fields"):
            field_mapper = FieldMapper()
            mapped_data = field_mapper.map(raw_data, structure_info)
            logger.info(f"Field mapping complete",
                       columns=len(mapped_data.columns) if hasattr(mapped_data, 'columns') else 'unknown')

        # Extract and normalize categories
        progress.update_task("Extracting categories", 60)
        with logger.context(operation="extract_categories"):
            category_extractor = CategoryExtractor()
            categorized_data = category_extractor.extract_categories(mapped_data)
            logger.info(f"Category extraction complete")

        # Normalize values
        progress.update_task("Normalizing values", 80)
        with logger.context(operation="normalize_values"):
            normalizer = ValueNormalizer()
            normalized_data = normalizer.normalize(categorized_data)
            logger.info(f"Value normalization complete",
                       rows=len(normalized_data) if hasattr(normalized_data, '__len__') else 'unknown',
                       columns=len(normalized_data.columns) if hasattr(normalized_data, 'columns') else 'unknown')

        # Validate output
        progress.update_task("Validating output", 90)
        with logger.context(operation="validate_output"):
            validated_data = validate_output(normalized_data)
            logger.info(f"Output validation complete")

        progress.complete_task("Processing complete")
        logger.info(f"File processing complete",
                   file_path=input_path,
                   output_format=output_format)

        return validated_data, None

    except FileNotFoundError as e:
        logger.error(f"File not found: {input_path}", exc_info=True)
        progress.fail_task(f"File not found: {input_path}")
        return None, f"File not found: {input_path}"

    except PermissionError as e:
        logger.error(f"Permission denied: {input_path}", exc_info=True)
        progress.fail_task(f"Permission denied: {input_path}")
        return None, f"Permission denied: {input_path}"

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}", exc_info=True)
        progress.fail_task(f"Processing failed: {str(e)}")

        # Provide more helpful error messages based on exception type
        if "memory" in str(e).lower():
            error_msg = f"Out of memory: The file is too large to process. Try using a smaller file or increasing available memory."
        elif "encoding" in str(e).lower():
            error_msg = f"Encoding error: Could not detect the file encoding. Try specifying the encoding manually."
        elif "corrupt" in str(e).lower() or "invalid" in str(e).lower():
            error_msg = f"File corruption: The file appears to be corrupted or in an invalid format."
        else:
            error_msg = str(e)

        return None, error_msg

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
            client = LLMFactory.create_client()
            if hasattr(client, 'get_model_info'):
                app_info['llm_info'] = client.get_model_info()
        except Exception as e:
            logger.warning(f"Could not get LLM information: {str(e)}")

        # Log health check
        logger.info("Health check request",
                   remote_addr=request.remote_addr,
                   user_agent=request.headers.get('User-Agent', 'Unknown'))

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
    Upload and process catalog file

    This endpoint accepts a file upload and converts it to a standardized format.
    The file is processed through the standardization pipeline and returned in the requested format.

    Returns:
        The processed file as an attachment or an error response
    """
    temp_file_path = None
    output_path = None

    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            logger.warning("Upload attempt with no file provided")
            return jsonify({
                'error': 'No file provided',
                'details': 'Please include a file in your request'
            }), 400

        file = request.files['file']

        # Check if filename is empty
        if file.filename == '':
            logger.warning("Upload attempt with empty filename")
            return jsonify({
                'error': 'No file selected',
                'details': 'The uploaded file has no name'
            }), 400

        # Get output format
        output_format = request.form.get('format', 'csv')
        if output_format not in ['csv', 'excel', 'json']:
            logger.warning(f"Invalid output format requested: {output_format}")
            return jsonify({
                'error': 'Invalid output format',
                'details': f"Format '{output_format}' is not supported. Use one of: csv, excel, json",
                'supported_formats': ['csv', 'excel', 'json']
            }), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_file_path)

        logger.info(f"File uploaded: {filename}",
                   file_size=os.path.getsize(temp_file_path),
                   output_format=output_format)

        # Process file
        with logger.context(operation="process_file", filename=filename, output_format=output_format):
            data, error = process_file(temp_file_path, output_format)

        # Check for processing error
        if error:
            logger.error(f"Error processing file: {error}")
            return jsonify({
                'error': 'Processing failed',
                'details': error,
                'suggestions': [
                    'Check that the file format is supported',
                    'Ensure the file is not corrupted',
                    'Try a different file format'
                ]
            }), 500

        # Create output file
        output_fd, output_path = tempfile.mkstemp(suffix=f'.{output_format}')
        os.close(output_fd)

        # Convert data to requested format
        try:
            if output_format == 'csv':
                data.to_csv(output_path, index=False)
            elif output_format == 'excel':
                data.to_excel(output_path, index=False)
            elif output_format == 'json':
                data.to_json(output_path, orient='records')

            logger.info(f"File processed successfully",
                       rows=len(data),
                       columns=len(data.columns),
                       output_format=output_format)

            # Return the file
            return send_file(
                output_path,
                as_attachment=True,
                download_name=f'standardized_catalog.{output_format}',
                mimetype='text/csv' if output_format == 'csv' else None
            )

        except Exception as e:
            logger.error(f"Error creating output file: {str(e)}", exc_info=True)
            return jsonify({
                'error': 'Output generation failed',
                'details': str(e)
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error in upload_file: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Server error',
            'details': str(e),
            'request_id': request.headers.get('X-Request-ID', 'unknown')
        }), 500

    finally:
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Temporary input file removed: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary input file: {str(e)}")

        # The output file will be automatically removed when the request is complete

@app.route('/api/analyze', methods=['POST'])
def analyze_file():
    """
    Analyze catalog file structure without full processing

    This endpoint accepts a file upload and analyzes its structure without performing
    the full standardization process. It returns information about the file structure,
    column types, and a sample of the data.

    Returns:
        JSON with file structure analysis or an error response
    """
    temp_file_path = None

    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            logger.warning("Analysis attempt with no file provided")
            return jsonify({
                'error': 'No file provided',
                'details': 'Please include a file in your request'
            }), 400

        file = request.files['file']

        # Check if filename is empty
        if file.filename == '':
            logger.warning("Analysis attempt with empty filename")
            return jsonify({
                'error': 'No file selected',
                'details': 'The uploaded file has no name'
            }), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_file_path)

        logger.info(f"File uploaded for analysis: {filename}",
                   file_size=os.path.getsize(temp_file_path))

        # Parse file
        with logger.context(operation="analyze_file", filename=filename):
            try:
                # Create parser based on file type
                parser = ParserFactory.create_parser(temp_file_path)
                logger.info(f"Parser created", parser_type=parser.__class__.__name__)

                # Parse the file
                raw_data = parser.parse()
                logger.info(f"File parsed successfully",
                           rows=len(raw_data) if hasattr(raw_data, '__len__') else 'unknown',
                           columns=len(raw_data.columns) if hasattr(raw_data, 'columns') else 'unknown')

                # Analyze structure
                analyzer = StructureAnalyzer()
                structure_info = analyzer.analyze(raw_data)
                logger.info(f"Structure analysis complete",
                           column_count=structure_info.get('column_count', 0))

                # Get sample data (limit to 10 rows and 100 columns for performance)
                if hasattr(raw_data, 'head'):
                    sample = raw_data.head(10)
                    # If there are too many columns, select a subset
                    if len(sample.columns) > 100:
                        sample = sample[sample.columns[:100]]
                    sample_data = sample.to_dict(orient='records')
                else:
                    sample_data = []

                # Return analysis
                analysis = {
                    'structure': structure_info,
                    'sample_data': sample_data,
                    'columns': list(raw_data.columns)[:100] if hasattr(raw_data, 'columns') else [],
                    'file_info': {
                        'filename': filename,
                        'size': os.path.getsize(temp_file_path),
                        'parser': parser.__class__.__name__
                    }
                }

                # Add LLM information if available
                try:
                    from core.llm.llm_factory import LLMFactory
                    client = LLMFactory.create_client()
                    if hasattr(client, 'get_model_info'):
                        analysis['llm_info'] = client.get_model_info()
                except Exception as e:
                    logger.warning(f"Could not get LLM information: {str(e)}")

                return jsonify(analysis)

            except FileNotFoundError:
                logger.error(f"File not found: {temp_file_path}")
                return jsonify({
                    'error': 'File not found',
                    'details': 'The uploaded file could not be found on the server'
                }), 404

            except PermissionError:
                logger.error(f"Permission denied: {temp_file_path}")
                return jsonify({
                    'error': 'Permission denied',
                    'details': 'The server does not have permission to read the file'
                }), 403

            except Exception as e:
                logger.error(f"Error analyzing file: {str(e)}", exc_info=True)
                return jsonify({
                    'error': 'Analysis failed',
                    'details': str(e),
                    'suggestions': [
                        'Check that the file format is supported',
                        'Ensure the file is not corrupted',
                        'Try a different file'
                    ]
                }), 500

    except Exception as e:
        logger.error(f"Unexpected error in analyze_file: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Server error',
            'details': str(e),
            'request_id': request.headers.get('X-Request-ID', 'unknown')
        }), 500

    finally:
        # Remove temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"Temporary file removed: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {str(e)}")

@app.route('/api/map-fields', methods=['POST'])
def map_fields():
    """
    Map fields from input columns to standardized format

    This endpoint accepts column names and sample data and returns suggested
    mappings to the standardized schema. It uses heuristics and pattern matching
    to determine the most likely mapping for each field.

    Returns:
        JSON with field mappings or an error response
    """
    try:
        # Validate request data
        if not request.is_json:
            logger.warning("Non-JSON request to map-fields endpoint")
            return jsonify({
                'error': 'Invalid content type',
                'details': 'Request must be application/json'
            }), 415

        data = request.json

        if not data:
            logger.warning("Empty request body")
            return jsonify({
                'error': 'Empty request',
                'details': 'Request body cannot be empty'
            }), 400

        # Check required fields
        missing_fields = []
        if 'columns' not in data:
            missing_fields.append('columns')
        if 'sample_data' not in data:
            missing_fields.append('sample_data')

        if missing_fields:
            logger.warning(f"Missing required fields: {', '.join(missing_fields)}")
            return jsonify({
                'error': 'Missing required fields',
                'details': f"The following fields are required: {', '.join(missing_fields)}",
                'required_fields': ['columns', 'sample_data']
            }), 400

        # Validate columns
        columns = data['columns']
        if not isinstance(columns, list):
            logger.warning("Invalid columns format")
            return jsonify({
                'error': 'Invalid columns format',
                'details': 'The columns field must be an array of strings'
            }), 400

        if not columns:
            logger.warning("Empty columns list")
            return jsonify({
                'error': 'Empty columns list',
                'details': 'The columns list cannot be empty'
            }), 400

        # Validate sample data
        sample_data = data['sample_data']
        if not isinstance(sample_data, list):
            logger.warning("Invalid sample_data format")
            return jsonify({
                'error': 'Invalid sample_data format',
                'details': 'The sample_data field must be an array of objects'
            }), 400

        if not sample_data:
            logger.warning("Empty sample_data list")
            return jsonify({
                'error': 'Empty sample_data list',
                'details': 'The sample_data list cannot be empty'
            }), 400

        # Log request details
        logger.info(f"Field mapping requested",
                   column_count=len(columns),
                   sample_row_count=len(sample_data))

        # Create field mapper
        with logger.context(operation="map_fields"):
            field_mapper = FieldMapper()

            # Convert sample data to format expected by mapper
            try:
                sample_rows = [[row.get(col) for col in columns] for row in sample_data]

                # Map fields
                mapping_results = field_mapper.map_fields(columns, sample_rows)

                # Log results
                logger.info(f"Field mapping complete",
                           mapping_count=len(mapping_results.get('mappings', [])))

                # Add additional information to the response
                mapping_results['request_info'] = {
                    'column_count': len(columns),
                    'sample_row_count': len(sample_data),
                    'timestamp': time.time(),
                    'llm_stats': field_mapper.llm_client.get_stats() if hasattr(field_mapper.llm_client, 'get_stats') else {}
                }

                return jsonify(mapping_results)

            except ValueError as e:
                logger.error(f"Value error in field mapping: {str(e)}")
                return jsonify({
                    'error': 'Invalid data format',
                    'details': str(e)
                }), 400

            except Exception as e:
                logger.error(f"Error mapping fields: {str(e)}", exc_info=True)
                return jsonify({
                    'error': 'Mapping failed',
                    'details': str(e)
                }), 500

    except Exception as e:
        logger.error(f"Unexpected error in map_fields: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'Server error',
            'details': str(e),
            'request_id': request.headers.get('X-Request-ID', 'unknown')
        }), 500

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='AV Catalog Standardizer')
    parser.add_argument('--input', '-i', type=str, help='Input file path')
    parser.add_argument('--output', '-o', type=str, help='Output file path')
    parser.add_argument('--format', '-f', type=str, default='csv',
                        choices=['csv', 'excel'], help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--api', '-a', action='store_true', help='Run as API server')
    parser.add_argument('--port', '-p', type=int, default=8080, help='API server port')
    return parser.parse_args()

def main():
    """Main application entry point"""
    # Parse arguments
    args = parse_arguments()

    # Run as API server if requested
    if args.api:
        logger.info(f"Starting API server on port {args.port}")
        app.run(host='0.0.0.0', port=args.port, debug=False)
        return 0

    # Ensure input file is provided for CLI mode
    if not args.input:
        logger.error("Input file is required when not running as API server")
        return 1

    # Setup logging for CLI mode
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)

    # Create output path if not provided
    output_path = args.output
    if not output_path:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_standardized.{args.format}")

    # Process file
    data, error = process_file(args.input, args.format)

    if error:
        logger.error(f"Processing failed: {error}")
        return 1

    # Write to output file
    if args.format == 'csv':
        data.to_csv(output_path, index=False)
    else:
        data.to_excel(output_path, index=False)

    logger.info(f"Standardization complete. Output saved to: {output_path}")
    return 0

if __name__ == "__main__":
    exit(main())