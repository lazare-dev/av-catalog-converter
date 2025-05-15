"""
Frontend Logging API Controller

Handles receiving and processing logs from the frontend application.
"""
import json
import logging
import time
from datetime import datetime
from flask import Blueprint, request, jsonify
from utils.logging.logger import Logger

# Create blueprint for logging API
logging_api = Blueprint('logging_api', __name__)

# Get logger
logger = Logger.get_logger(__name__)

@logging_api.route('/logs', methods=['POST'])
def receive_logs():
    """
    Receive logs from the frontend application
    
    This endpoint accepts log entries from the frontend and forwards them
    to the backend logging system with appropriate context.
    
    Returns:
        JSON response indicating success or failure
    """
    try:
        # Validate request
        if not request.is_json:
            logger.warning("Non-JSON request to logs endpoint")
            return jsonify({
                'error': 'Invalid content type',
                'details': 'Request must be application/json'
            }), 415
        
        data = request.json
        
        # Validate required fields
        if 'logs' not in data or not isinstance(data['logs'], list):
            logger.warning("Invalid logs format in request")
            return jsonify({
                'error': 'Invalid logs format',
                'details': 'Request must contain a "logs" array'
            }), 400
        
        # Get session information
        session_id = data.get('sessionId', 'unknown')
        user_agent = data.get('userAgent', 'unknown')
        client_timestamp = data.get('timestamp', datetime.now().isoformat())
        
        # Process each log entry
        processed_count = 0
        for entry in data['logs']:
            # Validate log entry
            if not all(k in entry for k in ['timestamp', 'level', 'component', 'message']):
                logger.warning(f"Invalid log entry format: {entry}")
                continue
            
            # Map frontend log level to Python log level
            level_map = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL
            }
            level = level_map.get(entry['level'], logging.INFO)
            
            # Extract log details
            timestamp = entry['timestamp']
            component = entry['component']
            message = entry['message']
            context = entry.get('context', {})
            
            # Add frontend context
            frontend_context = {
                'source': 'frontend',
                'session_id': session_id,
                'user_agent': user_agent,
                'component': component,
                'client_timestamp': timestamp,
                'server_timestamp': datetime.now().isoformat(),
                'time_difference_ms': calculate_time_difference(timestamp)
            }
            
            # Add user ID if available
            if 'userId' in entry and entry['userId']:
                frontend_context['user_id'] = entry['userId']
            
            # Merge contexts
            full_context = {**frontend_context, **context}
            
            # Log the entry with the backend logger
            with logger.context(**full_context):
                logger._log(level, f"[FRONTEND] {message}", **full_context)
            
            processed_count += 1
        
        # Log summary
        logger.info(f"Processed {processed_count} frontend log entries", 
                   session_id=session_id,
                   total_entries=len(data['logs']))
        
        # Return success response
        return jsonify({
            'success': True,
            'processed': processed_count,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        # Log error
        logger.error(f"Error processing frontend logs: {str(e)}", exc_info=True)
        
        # Return error response
        return jsonify({
            'error': 'Server error',
            'details': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

def calculate_time_difference(client_timestamp):
    """
    Calculate time difference between client and server in milliseconds
    
    Args:
        client_timestamp: ISO format timestamp from client
        
    Returns:
        Time difference in milliseconds (positive if server ahead, negative if client ahead)
    """
    try:
        # Parse client timestamp
        client_time = datetime.fromisoformat(client_timestamp.replace('Z', '+00:00'))
        
        # Get current server time
        server_time = datetime.now()
        
        # Calculate difference in milliseconds
        diff_ms = (server_time - client_time).total_seconds() * 1000
        
        return diff_ms
    except Exception:
        return 0
