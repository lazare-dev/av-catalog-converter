"""Validation utilities for data processing"""

import logging
import pandas as pd
from typing import Dict, Any, List

from config.schema import FIELD_ORDER, REQUIRED_FIELDS, SCHEMA_DICT, NUMERIC_FIELDS

logger = logging.getLogger(__name__)

def validate_output(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and prepare final output data

    Args:
        data (pd.DataFrame): Processed data

    Returns:
        pd.DataFrame: Validated data with correct structure
    """
    logger.info("Validating output data")

    # Create a new DataFrame with the correct column order
    result = pd.DataFrame(columns=FIELD_ORDER)

    # Copy data from input DataFrame, handling missing columns
    for field in FIELD_ORDER:
        if field in data.columns:
            result[field] = data[field]
        else:
            # Use default value from schema if available
            default = SCHEMA_DICT[field].default_value
            result[field] = default
            logger.warning(f"Field '{field}' not found in data, using default value: {default}")

    # Check required fields
    missing_required = [field for field in REQUIRED_FIELDS if field in result.columns and result[field].isna().any()]
    if missing_required:
        logger.warning(f"Missing values in required fields: {missing_required}")

        # For each required field with missing values, log the rows
        for field in missing_required:
            missing_rows = result[result[field].isna()].index.tolist()
            logger.warning(f"Field '{field}' has missing values in rows: {missing_rows[:10]}{'...' if len(missing_rows) > 10 else ''}")

    # Normalize numeric fields
    for field in NUMERIC_FIELDS:
        if field in result.columns:
            # First clean the data - remove currency symbols, commas, etc.
            if result[field].dtype == 'object':
                result[field] = result[field].astype(str).str.replace(r'[^\d.-]', '', regex=True)

            # Convert to numeric, forcing errors to NaN
            result[field] = pd.to_numeric(result[field], errors='coerce')

            # Fill NaN values with default
            if result[field].isna().any():
                missing_count = result[field].isna().sum()
                logger.warning(f"Field '{field}' has {missing_count} non-numeric values, replacing with 0")
                result[field] = result[field].fillna(0.0)

            # Format to 2 decimal places
            result[field] = result[field].round(2)

    # Normalize boolean fields
    if "Discontinued" in result.columns:
        # Map various values to Yes/No
        yes_values = ['yes', 'y', 'true', '1', 't', 'discontinued', 'obsolete', 'eol']
        result["Discontinued"] = result["Discontinued"].astype(str).str.lower()
        result["Discontinued"] = result["Discontinued"].apply(
            lambda x: "Yes" if x in yes_values else "No"
        )

    logger.info(f"Validation complete. Output has {len(result)} rows and {len(result.columns)} columns")
    return result

def validate_mapping(mapping: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Validate field mapping configuration

    Args:
        mapping (Dict[str, Any]): Field mapping configuration

    Returns:
        Dict[str, List[str]]: Validation results with any issues
    """
    issues = {
        "missing_required": [],
        "unknown_fields": [],
        "duplicate_mappings": []
    }

    # Log the mapping for debugging
    logger.info(f"Validating mapping: {mapping}")

    # Handle empty mapping
    if not mapping:
        logger.warning("Empty mapping provided for validation")
        for required in REQUIRED_FIELDS:
            issues["missing_required"].append(required)
        return issues

    # Check if mapping is in the correct format (target_field -> source_column)
    # or if it's inverted (source_column -> target_field)
    is_inverted = False
    for key in mapping.keys():
        if key in FIELD_ORDER:
            # This is the correct format (target_field -> source_column)
            is_inverted = False
            break
        # Check if any values are in FIELD_ORDER, which would indicate inverted mapping
        for value in mapping.values():
            if value in FIELD_ORDER:
                is_inverted = True
                break
        if is_inverted:
            break

    # If the mapping is inverted, convert it to the correct format
    if is_inverted:
        logger.info("Mapping appears to be inverted (source_column -> target_field), converting to target_field -> source_column")
        converted_mapping = {}
        for source, target in mapping.items():
            if target in FIELD_ORDER:
                converted_mapping[target] = source
        mapping = converted_mapping
        logger.info(f"Converted mapping: {mapping}")

    # Check for missing required fields
    mapped_fields = set(mapping.keys())  # The keys are the target fields, values are source columns
    for required in REQUIRED_FIELDS:
        if required not in mapped_fields:
            issues["missing_required"].append(required)

    # Check for unknown target fields
    valid_fields = set(FIELD_ORDER)
    for target in mapped_fields:
        if target not in valid_fields:
            issues["unknown_fields"].append(target)

    # Check for duplicate mappings (same source column mapped to multiple target fields)
    source_to_target = {}
    for target, source in mapping.items():
        if source in source_to_target:
            issues["duplicate_mappings"].append(f"{source} is mapped to both {target} and {source_to_target[source]}")
        else:
            source_to_target[source] = target

    # Log validation results
    logger.info(f"Validation results: {issues}")

    return issues
