# prompts/structure_analysis.py
"""
Structure analysis prompt builder
"""
import json
from prompts.templates.structure_template import STRUCTURE_ANALYSIS_TEMPLATE

def get_structure_analysis_prompt(data_sample: str, column_info: dict, 
                                header_info: dict, data_quality: dict) -> str:
    """
    Generate structure analysis prompt
    
    Args:
        data_sample (str): Sample data string
        column_info (dict): Column types and characteristics
        header_info (dict): Header detection information
        data_quality (dict): Data quality information
        
    Returns:
        str: Formatted prompt
    """
    # Format column information
    column_info_str = "Column information:\n"
    for col, info in column_info.items():
        sample_values = ', '.join([str(s) for s in info.get('samples', [])[:3]])
        column_info_str += f"- {col} (type: {info.get('type', 'unknown')}, uniqueness: {info.get('unique_ratio', 0):.2f})\n"
        column_info_str += f"  Sample values: {sample_values}\n"
    
    # Prepare prompt with all information
    prompt = STRUCTURE_ANALYSIS_TEMPLATE.format(
        data_sample=data_sample,
        column_info=column_info_str
    )
    
    return prompt
