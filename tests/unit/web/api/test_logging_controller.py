"""
Unit tests for the frontend logging API controller
"""
import json
import unittest
from unittest.mock import patch, MagicMock
from flask import Flask
from web.api.logging_controller import logging_api, receive_logs

class TestLoggingController(unittest.TestCase):
    """Test cases for the logging controller"""

    def setUp(self):
        """Set up test app"""
        self.app = Flask(__name__)
        self.app.register_blueprint(logging_api)
        self.client = self.app.test_client()
        
    @patch('web.api.logging_controller.logger')
    def test_receive_logs_valid_request(self, mock_logger):
        """Test receiving valid log entries"""
        # Create mock logger
        mock_context = MagicMock()
        mock_logger.context.return_value = mock_context
        mock_context.__enter__ = MagicMock()
        mock_context.__exit__ = MagicMock()
        
        # Create test data
        test_data = {
            'logs': [
                {
                    'timestamp': '2025-05-12T12:00:00.000Z',
                    'level': 'INFO',
                    'component': 'TestComponent',
                    'message': 'Test message',
                    'context': {'key': 'value'}
                },
                {
                    'timestamp': '2025-05-12T12:01:00.000Z',
                    'level': 'ERROR',
                    'component': 'TestComponent',
                    'message': 'Error message',
                    'context': {'error': 'test error'}
                }
            ],
            'sessionId': 'test-session-id',
            'timestamp': '2025-05-12T12:02:00.000Z',
            'userAgent': 'Test User Agent'
        }
        
        # Make request
        response = self.client.post(
            '/logs',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        self.assertEqual(response_data['processed'], 2)
        
        # Check that logger was called correctly
        self.assertEqual(mock_logger.context.call_count, 2)
        self.assertEqual(mock_logger._log.call_count, 2)
        
        # Check first log call
        first_call_args = mock_logger._log.call_args_list[0]
        self.assertEqual(first_call_args[0][0], 20)  # INFO level
        self.assertEqual(first_call_args[0][1], '[FRONTEND] Test message')
        
        # Check second log call
        second_call_args = mock_logger._log.call_args_list[1]
        self.assertEqual(second_call_args[0][0], 40)  # ERROR level
        self.assertEqual(second_call_args[0][1], '[FRONTEND] Error message')
        
        # Check that summary was logged
        mock_logger.info.assert_called_with(
            'Processed 2 frontend log entries',
            session_id='test-session-id',
            total_entries=2
        )
    
    @patch('web.api.logging_controller.logger')
    def test_receive_logs_invalid_content_type(self, mock_logger):
        """Test receiving logs with invalid content type"""
        # Make request with wrong content type
        response = self.client.post(
            '/logs',
            data='not json',
            content_type='text/plain'
        )
        
        # Check response
        self.assertEqual(response.status_code, 415)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['error'], 'Invalid content type')
        
        # Check that warning was logged
        mock_logger.warning.assert_called_with('Non-JSON request to logs endpoint')
    
    @patch('web.api.logging_controller.logger')
    def test_receive_logs_missing_logs_field(self, mock_logger):
        """Test receiving logs with missing logs field"""
        # Create test data with missing logs field
        test_data = {
            'sessionId': 'test-session-id',
            'timestamp': '2025-05-12T12:02:00.000Z'
        }
        
        # Make request
        response = self.client.post(
            '/logs',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        # Check response
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['error'], 'Invalid logs format')
        
        # Check that warning was logged
        mock_logger.warning.assert_called_with('Invalid logs format in request')
    
    @patch('web.api.logging_controller.logger')
    def test_receive_logs_invalid_log_entry(self, mock_logger):
        """Test receiving logs with invalid log entry"""
        # Create test data with invalid log entry
        test_data = {
            'logs': [
                {
                    'timestamp': '2025-05-12T12:00:00.000Z',
                    'level': 'INFO',
                    'component': 'TestComponent',
                    'message': 'Valid message'
                },
                {
                    # Missing required fields
                    'timestamp': '2025-05-12T12:01:00.000Z',
                    'level': 'ERROR'
                }
            ],
            'sessionId': 'test-session-id'
        }
        
        # Make request
        response = self.client.post(
            '/logs',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        # Check response
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.data)
        self.assertTrue(response_data['success'])
        self.assertEqual(response_data['processed'], 1)  # Only one valid log
        
        # Check that warning was logged for invalid entry
        mock_logger.warning.assert_called_with(
            "Invalid log entry format: {'timestamp': '2025-05-12T12:01:00.000Z', 'level': 'ERROR'}"
        )
        
        # Check that only one log was processed
        self.assertEqual(mock_logger._log.call_count, 1)
    
    @patch('web.api.logging_controller.logger')
    def test_receive_logs_server_error(self, mock_logger):
        """Test server error handling"""
        # Make logger raise an exception
        mock_logger._log.side_effect = Exception('Test exception')
        
        # Create test data
        test_data = {
            'logs': [
                {
                    'timestamp': '2025-05-12T12:00:00.000Z',
                    'level': 'INFO',
                    'component': 'TestComponent',
                    'message': 'Test message'
                }
            ],
            'sessionId': 'test-session-id'
        }
        
        # Make request
        response = self.client.post(
            '/logs',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        # Check response
        self.assertEqual(response.status_code, 500)
        response_data = json.loads(response.data)
        self.assertEqual(response_data['error'], 'Server error')
        self.assertEqual(response_data['details'], 'Test exception')
        
        # Check that error was logged
        mock_logger.error.assert_called_with(
            'Error processing frontend logs: Test exception',
            exc_info=True
        )

if __name__ == '__main__':
    unittest.main()
