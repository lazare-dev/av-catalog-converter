# prompts/field_mapping.py
"""
Field mapping prompt builder
"""
import json
from prompts.templates.mapping_template import FIELD_MAPPING_TEMPLATE

def get_field_mapping_prompt(standard_fields: dict, input_columns: list, 
                           column_samples: dict, structure_info: dict) -> str:
    """
    Generate field mapping prompt
    
    Args:
        standard_fields (dict): Standard field definitions to map to
        input_columns (list): Input column names to map from
        column_samples (dict): Sample values for each input column
        structure_info (dict): Structure analysis information
        
    Returns:
        str: Formatted prompt
    """
    # Format standard fields information
    standard_fields_str = "Standard fields to map:\n"
    for field_name, field_info in standard_fields.items():
        standard_fields_str += f"- {field_name}: {field_info.get('description', '')}\n"
        standard_fields_str += f"  Examples: {', '.join(field_info.get('examples', [])[:2])}\n"
        if field_info.get('mapping_hints'):
            hints = field_info.get('mapping_hints', [])
            if len(hints) > 3:
                hints = hints[:3]
            standard_fields_str += f"  Hints: {', '.join(hints)}\n"
    
    # Format input columns
    input_columns_str = "Input columns:\n"
    for col in input_columns:
        input_columns_str += f"- {col}\n"
    
    # Format column samples
    column_samples_str = "Column samples:\n"
    for col, samples in column_samples.items():
        samples_str = ', '.join([str(s) for s in samples[:3]])
        column_samples_str += f"- {col}: {samples_str}\n"
    
    # Format structure info
    structure_info_str = "Structure information:\n"
    # Include key details from structure analysis that help with mapping
    if 'column_analysis' in structure_info:
        for col, analysis in structure_info.get('column_analysis', {}).items():
            structure_info_str += f"- {col}: {analysis.get('purpose', '')}\n"
    else:
        structure_info_str += "No detailed structure analysis available.\n"
    
    # Prepare prompt with all information
    prompt = FIELD_MAPPING_TEMPLATE.format(
        standard_fields=standard_fields_str,
        input_columns=input_columns_str,
        column_samples=column_samples_str,
        structure_info=structure_info_str
    )
    
    return prompt