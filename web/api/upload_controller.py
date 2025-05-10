# web/api/upload_controller.py
"""
File upload handling
"""
import logging
import os
from pathlib import Path
from typing import Dict, Any
import tempfile

from flask import request, jsonify, Blueprint
from werkzeug.utils import secure_filename

from core.file_parser.parser_factory import ParserFactory

logger = logging.getLogger(__name__)

class UploadController:
    """Controller for file upload operations"""
    
    def __init__(self, upload_dir: str = None):
        """
        Initialize the upload controller
        
        Args:
            upload_dir (str, optional): Directory for uploaded files
        """
        self.upload_dir = upload_dir or os.path.join(tempfile.gettempdir(), 'av_catalog_uploads')
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def handle_upload(self, request):
        """
        Handle file upload request
        
        Args:
            request: HTTP request
            
        Returns:
            tuple: (response_data, status_code)
        """
        if 'file' not in request.files:
            return {'error': 'No file part'}, 400
            
        file = request.files['file']
        
        if file.filename == '':
            return {'error': 'No selected file'}, 400
            
        # Secure filename and save
        filename = secure_filename(file.filename)
        file_path = os.path.join(self.upload_dir, filename)
        file.save(file_path)
        
        logger.info(f"File uploaded: {filename}")
        
        # Try to parse the file to verify it's valid
        try:
            parser = ParserFactory.create_parser(file_path)
            sample = parser.get_sample(5)
            
            # Generate job ID
            import uuid
            job_id = str(uuid.uuid4())
            
            return {
                'success': True,
                'message': 'File uploaded successfully',
                'job_id': job_id,
                'filename': filename,
                'file_path': file_path,
                'sample_data': sample.to_dict() if not sample.empty else {}
            }, 200
        except Exception as e:
            logger.error(f"Error parsing uploaded file: {str(e)}")
            
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return {'error': f'Invalid file format: {str(e)}'}, 400
    
    def get_supported_formats(self):
        """
        Get supported file formats
        
        Returns:
            list: Supported file extensions
        """
        return list(ParserFactory.PARSER_MAP.keys())

    def upload_file(self, file):
        """
        Handle file upload and initial parsing
        
        Args:
            file: Uploaded file object
        
        Returns:
            Dict: Upload result with file info
        """
        file_path = None
        try:
            # Save file temporarily
            filename = secure_filename(file.filename)
            file_path = os.path.join(self.upload_folder, filename)
            file.save(file_path)
            
            # Parse file
            parser = ParserFactory.create_parser(file_path)
            data = parser.parse()
            
            # Get basic file info
            file_info = {
                'filename': filename,
                'path': file_path,
                'rows': len(data),
                'columns': len(data.columns),
                'column_names': list(data.columns)
            }
            
            return {'success': True, 'file_info': file_info}
        
        except Exception as e:
            logger.error(f"Error parsing uploaded file: {str(e)}")
            return {'error': f'Invalid file format: {str(e)}'}, 400
        
        finally:
            # Always clean up temporary files
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Temporary file removed: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file: {str(e)}")
