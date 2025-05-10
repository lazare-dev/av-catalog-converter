# web/api/mapping_controller.py
"""
Mapping review API
"""
import logging
from typing import Dict, Any, List
import pandas as pd

from services.mapping.field_mapper import FieldMapper
from services.structure.structure_analyzer import StructureAnalyzer
from core.file_parser.parser_factory import ParserFactory

logger = logging.getLogger(__name__)

class MappingController:
    """Controller for field mapping operations"""
    
    def __init__(self):
        """Initialize the mapping controller"""
        self.field_mapper = FieldMapper()
        self.structure_analyzer = StructureAnalyzer()
    
    def analyze_and_map(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze file structure and suggest field mappings
        
        Args:
            file_path (str): Path to input file
            
        Returns:
            Dict[str, Any]: Analysis and mapping results
        """
        try:
            # Parse file
            parser = ParserFactory.create_parser(file_path)
            data = parser.parse()
            
            # Analyze structure
            structure_info = self.structure_analyzer.analyze(data)
            
            # Extract potential field mappings
            field_mappings = structure_info.get('possible_field_mappings', {})
            
            return {
                'success': True,
                'field_mappings': field_mappings,
                'column_info': structure_info.get('column_analysis', {}),
                'data_quality': structure_info.get('data_quality_issues', []),
                'structure_info': structure_info
            }
            
        except Exception as e:
            logger.error(f"Error analyzing file: {str(e)}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def apply_mapping(self, file_path: str, user_mappings: Dict[str, str]) -> Dict[str, Any]:
        """
        Apply user-specified field mappings
        
        Args:
            file_path (str): Path to input file
            user_mappings (Dict[str, str]): User-specified mappings
            
        Returns:
            Dict[str, Any]: Mapping results
        """
        try:
            # Parse file
            parser = ParserFactory.create_parser(file_path)
            data = parser.parse()
            
            # Analyze structure
            structure_info = self.structure_analyzer.analyze(data)
            
            # Apply user mappings
            self.field_mapper.user_mappings = user_mappings
            mapped_data = self.field_mapper.map(data, structure_info)
            
            # Generate sample of mapped data
            sample = mapped_data.head(10).to_dict(orient='records')
            
            return {
                'success': True,
                'mapping_results': self.field_mapper.last_mapping_results,
                'sample_data': sample,
                'missing_required': self.field_mapper.last_mapping_results.get('missing_required', [])
            }
            
        except Exception as e:
            logger.error(f"Error applying mappings: {str(e)}")
            return {'error': f'Mapping failed: {str(e)}'}
    
    def get_mapping_suggestions(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get mapping suggestions for each target field
        
        Args:
            file_path (str): Path to input file
            
        Returns:
            Dict[str, List[Dict[str, Any]]]: Mapping suggestions
        """
        try:
            # Parse file
            parser = ParserFactory.create_parser(file_path)
            data = parser.parse()
            
            # Analyze structure
            structure_info = self.structure_analyzer.analyze(data)
            
            # Get suggestions from field mapper
            suggestions = self.field_mapper.get_mapping_suggestions(data.columns, structure_info)
            
            return {
                'success': True,
                'suggestions': suggestions
            }
            
        except Exception as e:
            logger.error(f"Error generating mapping suggestions: {str(e)}")
            return {'error': f'Failed to generate suggestions: {str(e)}'}
