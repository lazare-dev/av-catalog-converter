"""Field mapping service for standardizing catalog data"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import re
import json

from config.schema import OUTPUT_SCHEMA, FIELD_ORDER, REQUIRED_FIELDS, SCHEMA_DICT
from utils.parsers.json_parser import JSONParser
from core.llm.llm_factory import LLMFactory
from services.mapping.direct_mapper import DirectMapper
from services.mapping.semantic_mapper import SemanticMapper
from services.mapping.pattern_mapper import PatternMapper
from prompts.field_mapping import get_field_mapping_prompt
from config.settings import MAPPING_THRESHOLDS

# Configure logger
logger = logging.getLogger(__name__)

class FieldMapper:
    """Service for mapping input fields to standardized schema"""

    def __init__(self):
        """Initialize the field mapper"""
        self.logger = logging.getLogger(__name__)
        self.direct_mapper = DirectMapper()
        self.semantic_mapper = SemanticMapper()
        self.pattern_mapper = PatternMapper()
        self.json_parser = JSONParser()
        self.llm_client = LLMFactory.create_client()

        # Store user-provided mappings
        self.user_mappings = {}

        # Store last mapping results
        self.last_mapping_results = {}

        # Known manufacturer-specific patterns
        self.manufacturer_patterns = self._load_manufacturer_patterns()

    def _load_manufacturer_patterns(self) -> Dict[str, Dict[str, str]]:
        """
        Load manufacturer-specific mapping patterns

        Returns:
            Dict[str, Dict[str, str]]: Mapping patterns by manufacturer
        """
        try:
            # Try to load from a patterns file
            import os
            pattern_file = os.path.join(os.path.dirname(__file__), 'manufacturer_patterns.json')
            if os.path.exists(pattern_file):
                with open(pattern_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load manufacturer patterns: {str(e)}")

        # Return default patterns if file not found
        return {
            "sony": {
                "model_pattern": r"[A-Z]{3}-[A-Z0-9]{5,7}",
                "sku_prefix": "SNY-"
            },
            "samsung": {
                "model_pattern": r"[A-Z]{2}[0-9]{2}[A-Z][0-9]{4,5}",
                "sku_prefix": "SAM-"
            },
            "lg": {
                "model_pattern": r"[0-9]{2}[A-Z]{2}[0-9]{2,3}[A-Z]{1,2}",
                "sku_prefix": "LG-"
            },
            # Add more manufacturers as needed
        }

    def map(self, data: pd.DataFrame, structure_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Map input data to standardized schema

        Args:
            data (pd.DataFrame): Input data
            structure_info (Dict[str, Any]): Structure analysis results

        Returns:
            pd.DataFrame: Data with standardized fields
        """
        logger.info(f"Starting field mapping for {len(data)} rows")

        # Detect manufacturer if possible for manufacturer-specific patterns
        manufacturer = self._detect_manufacturer(data)
        logger.info(f"Detected manufacturer: {manufacturer}")

        # Multi-stage mapping process
        mapping_results = {}

        # Stage 1: Apply user-provided mappings if available
        if self.user_mappings:
            logger.info("Applying user-provided mappings")
            mapping_results = self.user_mappings.copy()

        # Stage 2: Direct header matching for unmapped fields
        if len(mapping_results) < len(FIELD_ORDER):
            logger.info("Applying direct header matching")
            direct_mappings = self.direct_mapper.map_fields(
                data.columns,
                already_mapped=list(mapping_results.values())
            )

            # Add direct mappings that don't conflict with user mappings
            for source, target in direct_mappings.items():
                if target not in mapping_results.values():
                    mapping_results[source] = target

        # Stage 3: Pattern-based mapping for specific fields
        if len(mapping_results) < len(FIELD_ORDER):
            logger.info("Applying pattern-based mapping")
            pattern_mappings = self.pattern_mapper.map_fields(
                data,
                manufacturer=manufacturer,
                already_mapped=list(mapping_results.values())
            )

            # Add pattern mappings that don't conflict
            for source, target in pattern_mappings.items():
                if target not in mapping_results.values():
                    mapping_results[source] = target

        # Stage 4: AI-assisted semantic mapping for remaining fields
        unmapped_required = [f for f in REQUIRED_FIELDS if f not in mapping_results.values()]
        if unmapped_required or len(mapping_results) < len(FIELD_ORDER) * 0.7:  # If less than 70% mapped
            logger.info("Applying AI-assisted semantic mapping")

            # The semantic_mapper will handle sample data preparation internally
            logger.info(f"Using semantic mapper with {min(15, len(data))} sample rows")

            # Generate AI-assisted mappings
            semantic_mappings = self.semantic_mapper.map_fields(
                data,
                structure_info,
                already_mapped=list(mapping_results.values())
            )

            # Add semantic mappings that don't conflict
            # Prioritize mappings for required fields
            required_mappings = {}
            other_mappings = {}

            for source, target in semantic_mappings.items():
                if target in REQUIRED_FIELDS:
                    required_mappings[source] = target
                else:
                    other_mappings[source] = target

            # First add required field mappings
            for source, target in required_mappings.items():
                if target not in mapping_results.values():
                    mapping_results[source] = target
                    logger.info(f"Added required field mapping: {source} -> {target}")

            # Then add other mappings
            for source, target in other_mappings.items():
                if target not in mapping_results.values():
                    mapping_results[source] = target

        # Apply the mappings to create the standardized dataframe
        standardized_df = self._apply_mappings_dict(data, mapping_results)

        # Store the last mapping results for reference
        self.last_mapping_results = mapping_results

        return standardized_df

    def _detect_manufacturer(self, data: pd.DataFrame) -> Optional[str]:
        """
        Detect the manufacturer from the data

        Args:
            data (pd.DataFrame): Input data

        Returns:
            Optional[str]: Detected manufacturer name or None
        """
        # Check if there's a manufacturer column
        manufacturer_cols = [col for col in data.columns if 'manufacturer' in col.lower()
                            or 'brand' in col.lower() or 'vendor' in col.lower()]

        if manufacturer_cols:
            # Get the most common value in the first manufacturer column
            mfr_col = manufacturer_cols[0]
            mfr_counts = data[mfr_col].value_counts()
            if not mfr_counts.empty:
                manufacturer = mfr_counts.index[0]
                if isinstance(manufacturer, str) and len(manufacturer) > 0:
                    return manufacturer.lower()

        return None

    def map_fields(self, columns: List[str], sample_rows: List[List[Any]]) -> Dict[str, Any]:
        """
        Map fields using AI assistance

        Args:
            columns (List[str]): Column names
            sample_rows (List[List[Any]]): Sample data rows

        Returns:
            Dict[str, Any]: Mapping results with confidence scores
        """
        logger.info(f"Mapping {len(columns)} fields with {len(sample_rows)} sample rows")

        # Convert sample rows to a more readable format
        sample_data = []
        for row in sample_rows:
            sample_data.append({col: val for col, val in zip(columns, row)})

        # Generate prompt for the LLM
        # Create an empty structure_info dict if we don't have one
        structure_info = {"columns": {}}

        # Convert schema fields to the expected format
        standard_fields = {field: {"description": "Standard field", "examples": []} for field in FIELD_ORDER}

        # Convert sample data to column samples format
        column_samples = {}
        for col in columns:
            column_samples[col] = [row.get(col, "") for row in sample_data if row.get(col) is not None][:5]

        prompt = get_field_mapping_prompt(standard_fields, columns, column_samples, structure_info)

        # Get mapping suggestions from LLM
        try:
            response = self.llm_client.generate_response(prompt)

            # Parse the JSON response
            mapping_data = self.json_parser.parse_json(response)

            # Validate the response format
            if not isinstance(mapping_data, dict):
                logger.warning("Invalid mapping response format - not a dict")
                mapping_data = {'field_mappings': {}, 'notes': 'Failed to parse LLM response'}
            elif 'field_mappings' not in mapping_data and 'mappings' in mapping_data:
                # Handle old format
                logger.info("Converting old mappings format to new field_mappings format")
                mapping_data['field_mappings'] = mapping_data['mappings']
                del mapping_data['mappings']
            elif 'field_mappings' not in mapping_data:
                # Try to extract mappings from the response directly
                try:
                    direct_mappings = {}
                    # Look for patterns like "input_column_name": "Target_Field_Name"
                    pattern = r'"([^"]+)"\s*:\s*"([^"]+)"'
                    matches = re.findall(pattern, response)

                    for source, target in matches:
                        if source in columns and target in FIELD_ORDER:
                            direct_mappings[source] = target

                    if direct_mappings:
                        logger.info(f"Extracted {len(direct_mappings)} mappings directly from response")
                        mapping_data = {
                            'field_mappings': direct_mappings,
                            'notes': 'Mappings extracted directly from response'
                        }
                    else:
                        logger.warning("Invalid mapping response format - no mappings")
                        mapping_data = {
                            'field_mappings': {},
                            'notes': 'Failed to parse LLM response'
                        }
                except Exception as e:
                    logger.warning(f"Failed to extract mappings directly: {str(e)}")
                    mapping_data = {
                        'field_mappings': {},
                        'notes': 'Failed to parse LLM response'
                    }

            # Convert field_mappings to the expected format for the API
            if 'field_mappings' in mapping_data and isinstance(mapping_data['field_mappings'], dict):
                mappings_list = []
                for source, target in mapping_data['field_mappings'].items():
                    # Handle both string targets and dict targets with confidence
                    if isinstance(target, dict) and 'column' in target:
                        mappings_list.append({
                            'source_field': source,
                            'target_field': target['column'],
                            'confidence': target.get('confidence', 0.7),
                            'reasoning': target.get('reasoning', 'No reasoning provided')
                        })
                    elif isinstance(target, str):
                        mappings_list.append({
                            'source_field': source,
                            'target_field': target,
                            'confidence': 0.7,
                            'reasoning': 'Direct mapping'
                        })

                mapping_data['mappings'] = mappings_list

            return mapping_data

        except Exception as e:
            logger.error(f"Error parsing mapping response: {str(e)}")
            return {
                'mappings': [],
                'notes': f'Error: {str(e)}',
                'raw_response': response
            }

    def _apply_mappings(self, data: pd.DataFrame, mappings: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply field mappings to a DataFrame

        Args:
            data (pd.DataFrame): Input data
            mappings (List[Dict[str, Any]]): Field mappings

        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        # Create a copy of the input data
        result = data.copy()

        # Create a mapping dictionary
        mapping_dict = {}
        for mapping in mappings:
            source = mapping.get('source_field')
            target = mapping.get('target_field')
            if source and target and source in data.columns:
                mapping_dict[source] = target

        # Apply the mappings
        result = self._apply_mappings_dict(result, mapping_dict)

        return result

    def _apply_mappings_dict(self, data: pd.DataFrame, mapping_dict: Dict[str, str]) -> pd.DataFrame:
        """
        Apply field mappings from a dictionary

        Args:
            data (pd.DataFrame): Input data
            mapping_dict (Dict[str, str]): Mapping from source to target fields

        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        # Create a copy of the input data
        result = data.copy()

        # Rename columns according to the mapping
        rename_dict = {src: tgt for src, tgt in mapping_dict.items() if src in result.columns}
        if rename_dict:
            result = result.rename(columns=rename_dict)

        # Add missing required fields
        for field in REQUIRED_FIELDS:
            if field not in result.columns:
                result[field] = ""

        # Ensure all schema fields are present
        for field in FIELD_ORDER:
            if field not in result.columns:
                result[field] = ""

        # Reorder columns according to schema
        ordered_cols = [col for col in FIELD_ORDER if col in result.columns]
        other_cols = [col for col in result.columns if col not in FIELD_ORDER]
        result = result[ordered_cols + other_cols]

        return result
