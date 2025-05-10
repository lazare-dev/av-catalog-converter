# web/api/preview_controller.py
"""
Data preview API
"""
import logging
from typing import Dict, Any, List
import pandas as pd
import json

from core.file_parser.parser_factory import ParserFactory

logger = logging.getLogger(__name__)

class PreviewController:
    """Controller for data preview operations"""
    
    def __init__(self):
        """Initialize the preview controller"""
        pass
    
    def get_raw_preview(self, file_path: str, rows: int = 10) -> Dict[str, Any]:
        """
        Get preview of raw input file
        
        Args:
            file_path (str): Path to input file
            rows (int): Number of rows to preview
            
        Returns:
            Dict[str, Any]: Preview data
        """
        try:
            # Parse file
            parser = ParserFactory.create_parser(file_path)
            sample = parser.get_sample(rows)
            
            # Convert to dict for JSON serialization
            preview = sample.to_dict(orient='records')
            columns = list(sample.columns)
            
            return {
                'success': True,
                'preview': preview,
                'columns': columns,
                'row_count': len(preview)
            }
            
        except Exception as e:
            logger.error(f"Error generating raw preview: {str(e)}")
            return {'error': f'Preview failed: {str(e)}'}
    
    def get_mapped_preview(self, output_file: str, rows: int = 20) -> Dict[str, Any]:
        """
        Get preview of mapped/processed output file
        
        Args:
            output_file (str): Path to output file
            rows (int): Number of rows to preview
            
        Returns:
            Dict[str, Any]: Preview data
        """
        try:
            # Load output file
            if output_file.endswith('.csv'):
                data = pd.read_csv(output_file)
            else:  # excel
                data = pd.read_excel(output_file)
                
            # Generate preview
            preview = data.head(rows).to_dict(orient='records')
            columns = list(data.columns)
            
            return {
                'success': True,
                'preview': preview,
                'columns': columns,
                'row_count': len(preview),
                'total_rows': len(data)
            }
            
        except Exception as e:
            logger.error(f"Error generating mapped preview: {str(e)}")
            return {'error': f'Preview failed: {str(e)}'}
    
    def get_column_statistics(self, file_path: str, columns: List[str] = None) -> Dict[str, Any]:
        """
        Get statistical information about columns
        
        Args:
            file_path (str): Path to input file
            columns (List[str], optional): Specific columns to analyze
            
        Returns:
            Dict[str, Any]: Column statistics
        """
        try:
            # Parse file
            parser = ParserFactory.create_parser(file_path)
            data = parser.parse()
            
            # Filter columns if specified
            if columns:
                data = data[columns]
                
            # Calculate statistics
            stats = {}
            
            for col in data.columns:
                col_data = data[col]
                
                # Basic stats
                col_stats = {
                    'count': len(col_data),
                    'missing': col_data.isna().sum(),
                    'unique': col_data.nunique()
                }
                
                # Numeric stats if applicable
                if pd.api.types.is_numeric_dtype(col_data):
                    col_stats.update({
                        'min': float(col_data.min()) if not col_data.isna().all() else None,
                        'max': float(col_data.max()) if not col_data.isna().all() else None,
                        'mean': float(col_data.mean()) if not col_data.isna().all() else None,
                        'median': float(col_data.median()) if not col_data.isna().all() else None
                    })
                
                # Text stats if applicable
                elif col_data.dtype == object:
                    # Length statistics
                    lengths = col_data.astype(str).str.len()
                    col_stats.update({
                        'min_length': int(lengths.min()) if not lengths.isna().all() else None,
                        'max_length': int(lengths.max()) if not lengths.isna().all() else None,
                        'mean_length': float(lengths.mean()) if not lengths.isna().all() else None
                    })
                    
                    # Most common values
                    value_counts = col_data.value_counts(dropna=False).head(5).to_dict()
                    col_stats['common_values'] = {str(k): int(v) for k, v in value_counts.items()}
                
                stats[col] = col_stats
                
            return {
                'success': True,
                'statistics': stats
            }
            
        except Exception as e:
            logger.error(f"Error generating column statistics: {str(e)}")
            return {'error': f'Statistics calculation failed: {str(e)}'}
