# web/api/validation_controller.py
"""
Validation controller for field mappings
"""
import logging
from typing import Dict, Any, List

from utils.helpers.validation_helpers import validate_mapping
from config.schema import FIELD_ORDER, REQUIRED_FIELDS

logger = logging.getLogger(__name__)

class ValidationController:
    """Controller for validation operations"""
    
    def __init__(self):
        """Initialize the validation controller"""
        pass
    
    def validate_field_mapping(self, mapping: Dict[str, str]) -> Dict[str, Any]:
        """
        Validate field mapping configuration
        
        Args:
            mapping (Dict[str, str]): Field mapping configuration
            
        Returns:
            Dict[str, Any]: Validation results
        """
        logger.info(f"Validating field mapping with {len(mapping)} mappings")
        
        # Validate the mapping
        issues = validate_mapping(mapping)
        
        # Check if there are any issues
        has_issues = any(len(issues[key]) > 0 for key in issues)
        
        # Log validation results
        if has_issues:
            logger.warning(f"Validation issues found in field mapping", issues=issues)
            for issue_type, issue_list in issues.items():
                if issue_list:
                    logger.warning(f"{issue_type}: {', '.join(issue_list)}")
        else:
            logger.info(f"Field mapping validation successful")
        
        return {
            'success': not has_issues,
            'issues': issues
        }
    
    def get_required_fields(self) -> List[str]:
        """
        Get list of required fields
        
        Returns:
            List[str]: Required fields
        """
        return REQUIRED_FIELDS
    
    def get_field_order(self) -> List[str]:
        """
        Get standard field order
        
        Returns:
            List[str]: Field order
        """
        return FIELD_ORDER
