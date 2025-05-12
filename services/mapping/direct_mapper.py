"""
Direct mapping strategies
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple
import re
from difflib import SequenceMatcher

from services.mapping.field_definitions import FIELD_DEFINITIONS
from config.settings import MAPPING_THRESHOLDS

class DirectMapper:
    """Maps fields based on exact or similar name matches"""

    def __init__(self):
        """Initialize the direct mapper"""
        self.logger = logging.getLogger(__name__)

    def map_fields(self, input_columns: List[str],
                 standard_fields: List[str] = None,
                 already_mapped: List[str] = None) -> Dict[str, str]:
        """
        Map fields based on direct name matching

        Args:
            input_columns (List[str]): Source input column names
            standard_fields (List[str], optional): Target standard field names
            already_mapped (List[str], optional): Fields that are already mapped

        Returns:
            Dict[str, str]: Mapping dict (input -> standard)
        """
        # For backward compatibility
        if standard_fields is None:
            from config.schema import FIELD_ORDER
            standard_fields = FIELD_ORDER

        if already_mapped is None:
            already_mapped = []

        # Filter out already mapped fields
        available_std_fields = [f for f in standard_fields if f not in already_mapped]

        # Call the original implementation
        mappings, _ = self._map_fields_internal(available_std_fields, input_columns)

        # Convert the mapping direction for the new interface
        result = {input_col: std_field for std_field, input_col in mappings.items()}

        return result

    def _map_fields_internal(self, standard_fields: List[str],
                 input_columns: List[str]) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Internal implementation of field mapping

        Args:
            standard_fields (List[str]): Target standard field names
            input_columns (List[str]): Source input column names

        Returns:
            Tuple[Dict[str, str], Dict[str, float]]:
                Mapping dict (standard -> input) and confidence scores
        """
        self.logger.info("Performing direct field mapping")

        mappings = {}
        confidence_scores = {}

        # Create normalized versions of input columns for matching
        normalized_columns = {self._normalize_name(col): col for col in input_columns}

        # First, try exact matches (case-insensitive)
        for std_field in standard_fields:
            normalized_std = self._normalize_name(std_field)

            # Try exact match
            if normalized_std in normalized_columns:
                orig_col = normalized_columns[normalized_std]
                mappings[std_field] = orig_col
                confidence_scores[std_field] = 1.0
                self.logger.debug(f"Exact match: {std_field} -> {orig_col}")

        # Remove already mapped fields
        remaining_std_fields = [f for f in standard_fields if f not in mappings]
        mapped_input_cols = set(mappings.values())
        remaining_input_cols = [c for c in input_columns if c not in mapped_input_cols]
        normalized_remaining = {self._normalize_name(col): col for col in remaining_input_cols}

        # Next, try token matches (field name contains standard field name or vice versa)
        for std_field in remaining_std_fields:
            normalized_std = self._normalize_name(std_field)
            std_tokens = set(normalized_std.split('_'))

            best_match = None
            best_score = 0

            for norm_col, orig_col in normalized_remaining.items():
                col_tokens = set(norm_col.split('_'))

                # Calculate token overlap ratio
                if len(std_tokens) == 0 or len(col_tokens) == 0:
                    continue

                intersection = std_tokens.intersection(col_tokens)

                # Score based on relative token overlap
                if intersection:
                    std_overlap = len(intersection) / len(std_tokens)
                    col_overlap = len(intersection) / len(col_tokens)

                    # Take geometric mean of the two overlap ratios
                    overlap_score = (std_overlap * col_overlap) ** 0.5

                    # Additional weight based on intersection size
                    weight = min(1.0, len(intersection) / 3)
                    weighted_score = overlap_score * (0.7 + 0.3 * weight)

                    if weighted_score > best_score and weighted_score >= MAPPING_THRESHOLDS["medium_confidence"]:
                        best_score = weighted_score
                        best_match = orig_col

            if best_match:
                mappings[std_field] = best_match
                confidence_scores[std_field] = best_score
                self.logger.debug(f"Token match: {std_field} -> {best_match} (score: {best_score:.2f})")

                # Remove this column from remaining
                norm_best = self._normalize_name(best_match)
                if norm_best in normalized_remaining:
                    del normalized_remaining[norm_best]

        # Remove already mapped fields again
        remaining_std_fields = [f for f in standard_fields if f not in mappings]
        mapped_input_cols = set(mappings.values())
        remaining_input_cols = [c for c in input_columns if c not in mapped_input_cols]
        normalized_remaining = {self._normalize_name(col): col for col in remaining_input_cols}

        # Try fuzzy matches for remaining fields
        for std_field in remaining_std_fields:
            field_hints = self._get_field_hint_terms(std_field)

            best_match = None
            best_score = 0

            for norm_col, orig_col in normalized_remaining.items():
                # Calculate fuzzy similarity
                similarity = SequenceMatcher(None, norm_col, self._normalize_name(std_field)).ratio()

                # Check for hint terms in the column name
                has_hint = any(hint.lower() in norm_col.lower() for hint in field_hints)

                # Adjust score based on hint presence
                adjusted_score = similarity * (1.2 if has_hint else 1.0)

                if adjusted_score > best_score and adjusted_score >= MAPPING_THRESHOLDS["low_confidence"]:
                    best_score = adjusted_score
                    best_match = orig_col

            if best_match:
                mappings[std_field] = best_match
                confidence_scores[std_field] = best_score
                self.logger.debug(f"Fuzzy match: {std_field} -> {best_match} (score: {best_score:.2f})")

                # Remove this column from remaining
                norm_best = self._normalize_name(best_match)
                if norm_best in normalized_remaining:
                    del normalized_remaining[norm_best]

        self.logger.info(f"Direct mapping identified {len(mappings)} fields")
        return mappings, confidence_scores

    def _normalize_name(self, name: str) -> str:
        """
        Normalize a field name for comparison

        Args:
            name (str): Field name to normalize

        Returns:
            str: Normalized field name
        """
        # Convert to lowercase
        normalized = str(name).lower()

        # Replace special characters and separators with underscores
        normalized = re.sub(r'[^a-z0-9]', '_', normalized)

        # Replace multiple underscores with a single one
        normalized = re.sub(r'_+', '_', normalized)

        # Remove leading/trailing underscores
        normalized = normalized.strip('_')

        return normalized

    def _get_field_hint_terms(self, std_field: str) -> List[str]:
        """
        Get hint terms for a standard field

        Args:
            std_field (str): Standard field name

        Returns:
            List[str]: List of hint terms
        """
        field_def = FIELD_DEFINITIONS.get(std_field)
        if not field_def:
            return []

        # Extract terms from field definition
        hints = []

        # Add the field name itself (split into parts)
        field_parts = re.findall(r'[A-Z][a-z]*', std_field)
        hints.extend([part.lower() for part in field_parts])

        # Add mapping hints from field definition
        for hint in field_def.mapping_hints:
            # Extract terms in quotes as exact match hints
            quoted_terms = re.findall(r"'([^']*)'", hint)
            hints.extend(quoted_terms)

            # Extract other keywords
            keywords = re.findall(r'\b([A-Za-z0-9]+)\b', hint)
            common_words = {'often', 'labeled', 'as', 'or', 'the', 'may', 'be', 'in', 'with', 'and', 'for', 'to', 'of'}
            hints.extend([kw.lower() for kw in keywords if kw.lower() not in common_words and len(kw) > 2])

        return list(set(hints))  # Deduplicate