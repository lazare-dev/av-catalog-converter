# web/api/export_controller.py
"""
Export functionality
"""
import logging
import os
from typing import Dict, Any
import pandas as pd

from utils.helpers.validation_helpers import validate_output

logger = logging.getLogger(__name__)

class ExportController:
    """Controller for data export operations"""
    
    def __init__(self):
        """Initialize the export controller"""
        pass
    
    def export_data(self, data: pd.DataFrame, output_path: str, format: str = 'csv') -> Dict[str, Any]:
        """
        Export data to file
        
        Args:
            data (pd.DataFrame): Data to export
            output_path (str): Export file path
            format (str): Export format (csv or excel)
            
        Returns:
            Dict[str, Any]: Export result
        """
        try:
            # Validate data before export
            validated_data = validate_output(data)
            
            # Export based on format
            if format.lower() == 'csv':
                validated_data.to_csv(output_path, index=False)
            elif format.lower() in ['excel', 'xlsx']:
                validated_data.to_excel(output_path, index=False)
            else:
                return {'error': f'Unsupported export format: {format}'}
                
            return {
                'success': True,
                'message': 'Data exported successfully',
                'output_path': output_path,
                'format': format,
                'rows': len(validated_data),
                'columns': len(validated_data.columns)
            }
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return {'error': f'Export failed: {str(e)}'}
    
    def get_export_formats(self) -> List[Dict[str, str]]:
        """
        Get supported export formats
        
        Returns:
            List[Dict[str, str]]: Supported formats with details
        """
        return [
            {
                'id': 'csv',
                'name': 'CSV (Comma Separated Values)',
                'extension': '.csv',
                'description': 'Standard text-based format compatible with most spreadsheet applications'
            },
            {
                'id': 'excel',
                'name': 'Excel Spreadsheet',
                'extension': '.xlsx',
                'description': 'Microsoft Excel format with formatting preserved'
            }
        ]
    
    def generate_filename(self, original_filename: str, format: str) -> str:
        """
        Generate export filename based on original file
        
        Args:
            original_filename (str): Original file name
            format (str): Export format
            
        Returns:
            str: Generated filename
        """
        # Get base name without extension
        base_name = os.path.splitext(os.path.basename(original_filename))[0]
        
        # Add suffix and extension
        if format.lower() == 'csv':
            return f"{base_name}_standardized.csv"
        else:  # excel
            return f"{base_name}_standardized.xlsx"