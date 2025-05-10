"""
Pattern-based mapping
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple
import re

from services.mapping.field_definitions import FIELD_DEFINITIONS
from config.settings import MAPPING_THRESHOLDS

class PatternMapper:
    """Maps fields based on data patterns and regular expressions"""
    
    def __init__(self):
        """Initialize the pattern mapper"""
        self.logger = logging.getLogger(__name__)
    
    def map_fields(self, standard_fields: List[str], input_columns: List[str],
                 data: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, float]]:
        """
        Map fields based on pattern recognition
        
        Args:
            standard_fields (List[str]): Target standard field names
            input_columns (List[str]): Source input column names
            data (pd.DataFrame): Input data for pattern analysis
            
        Returns:
            Tuple[Dict[str, str], Dict[str, float]]: 
                Mapping dict (standard -> input) and confidence scores
        """
        self.logger.info("Performing pattern-based field mapping")
        
        mappings = {}
        confidence_scores = {}
        
        # Apply field-specific pattern matchers
        for std_field in standard_fields:
            field_def = FIELD_DEFINITIONS.get(std_field)
            if not field_def:
                continue
                
            # Get pattern matcher function for this field type
            matcher_func = self._get_field_matcher(std_field)
            if not matcher_func:
                continue
                
            # Test each input column against the pattern
            best_match = None
            best_score = 0
            
            for col in input_columns:
                if col not in data.columns:
                    continue
                    
                # Get a sample of data
                sample = data[col].dropna().sample(min(100, len(data[col].dropna())))
                
                # Apply the matcher function
                score = matcher_func(sample, field_def)
                
                if score > best_score and score >= MAPPING_THRESHOLDS["low_confidence"]:
                    best_score = score
                    best_match = col
            
            if best_match:
                mappings[std_field] = best_match
                confidence_scores[std_field] = best_score
                self.logger.debug(f"Pattern match: {std_field} -> {best_match} (score: {best_score:.2f})")
                
                # Remove this column from further consideration
                if best_match in input_columns:
                    input_columns.remove(best_match)
        
        self.logger.info(f"Pattern mapping identified {len(mappings)} fields")
        return mappings, confidence_scores
    
    def _get_field_matcher(self, field_name: str):
        """
        Get the appropriate matcher function for a field
        
        Args:
            field_name (str): Standard field name
            
        Returns:
            function: Matcher function or None
        """
        # Mapping of field names to matcher functions
        matchers = {
            "SKU": self._match_sku,
            "Short_Description": self._match_short_description,
            "Long_Description": self._match_long_description,
            "Model": self._match_model,
            "Category_Group": self._match_category_group,
            "Category": self._match_category,
            "Manufacturer": self._match_manufacturer,
            "Manufacturer_SKU": self._match_manufacturer_sku,
            "Image_URL": self._match_image_url,
            "Document_URL": self._match_document_url,
            "Buy_Cost": self._match_buy_cost,
            "Trade_Price": self._match_trade_price,
            "MSRP_GBP": self._match_msrp_gbp,
            "MSRP_USD": self._match_msrp_usd,
            "MSRP_EUR": self._match_msrp_eur,
            "Discontinued": self._match_discontinued,
        }
        
        return matchers.get(field_name)
    
    def _match_sku(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match SKU patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        # SKUs are typically:
        # 1. Alphanumeric with possible separators like - or _
        # 2. Consistent format across products
        # 3. Unique per product
        
        if len(sample) == 0:
            return 0.0
            
        # Check uniqueness
        unique_ratio = len(sample.unique()) / len(sample)
        if unique_ratio < 0.9:  # SKUs should be highly unique
            return 0.0
            
        # Check for SKU-like patterns
        sample_str = sample.astype(str)
        
        # Apply patterns from field definition
        pattern_matches = 0
        for pattern in field_def.patterns:
            matches = sample_str.str.match(pattern)
            pattern_matches += matches.sum()
            
        pattern_score = pattern_matches / len(sample)
        
        # Check for general SKU characteristics
        sku_characteristics = sample_str.str.match(r'^[A-Za-z0-9\-_]{5,20}$')
        general_score = sku_characteristics.sum() / len(sample)
        
        # Check format consistency (length variance should be small)
        length_variance = sample_str.str.len().var()
        consistency_score = 1.0 if length_variance < 5 else 0.5
        
        # Combine scores
        final_score = (pattern_score * 0.4 + general_score * 0.4 + consistency_score * 0.2)
        
        return final_score
    
    def _match_short_description(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match Short Description patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        if len(sample) == 0:
            return 0.0
            
        sample_str = sample.astype(str)
        
        # Short descriptions typically:
        # 1. Have moderate length (15-100 chars)
        # 2. Contain product type words
        # 3. Don't contain very long text
        
        # Check length characteristics
        lengths = sample_str.str.len()
        avg_length = lengths.mean()
        
        # Ideal length for short descriptions (not too short, not too long)
        length_score = 0.0
        if 15 <= avg_length <= 100:
            length_score = 0.6
        elif 10 <= avg_length < 15 or 100 < avg_length <= 150:
            length_score = 0.3
            
        # Check for product type words
        product_types = ['projector', 'speaker', 'display', 'screen', 'cable', 'mount', 
                       'microphone', 'mixer', 'controller', 'processor', 'amplifier']
        
        type_matches = 0
        for word in product_types:
            matches = sample_str.str.contains(fr'\b{word}\b', case=False)
            type_matches += matches.sum()
            
        type_score = min(0.8, type_matches / len(sample))
        
        # Final score combines length and content characteristics
        final_score = (length_score * 0.6 + type_score * 0.4)
        
        return final_score
    
    def _match_long_description(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match Long Description patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        if len(sample) == 0:
            return 0.0
            
        sample_str = sample.astype(str)
        
        # Long descriptions typically:
        # 1. Have substantial length (>100 chars)
        # 2. Contain detailed product information
        # 3. Often include technical terms
        
        # Check length characteristics
        lengths = sample_str.str.len()
        avg_length = lengths.mean()
        
        # Score based on length (long descriptions should be substantial)
        length_score = 0.0
        if avg_length > 200:
            length_score = 0.8
        elif 100 <= avg_length <= 200:
            length_score = 0.5
        elif 50 <= avg_length < 100:
            length_score = 0.2
            
        # Check for technical terms and detailed info patterns
        tech_terms = ['resolution', 'specifications', 'features', 'dimensions', 'technology',
                    'compatible', 'input', 'output', 'warranty', 'included']
        
        term_matches = 0
        for term in tech_terms:
            matches = sample_str.str.contains(fr'\b{term}\b', case=False)
            term_matches += matches.sum()
            
        detail_score = min(0.8, term_matches / (len(sample) * 2))
        
        # Final score
        final_score = (length_score * 0.7 + detail_score * 0.3)
        
        return final_score
    
    def _match_model(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match Model Number patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        if len(sample) == 0:
            return 0.0
            
        sample_str = sample.astype(str)
        
        # Model numbers typically:
        # 1. Follow manufacturer-specific patterns
        # 2. Often include alphanumeric codes
        # 3. Higher uniqueness than categories but lower than SKUs
        
        # Check uniqueness
        unique_ratio = len(sample.unique()) / len(sample)
        uniqueness_score = 0.0
        if 0.7 <= unique_ratio <= 1.0:
            uniqueness_score = 0.6
        elif 0.4 <= unique_ratio < 0.7:
            uniqueness_score = 0.3
            
        # Apply patterns from field definition
        pattern_matches = 0
        for pattern in field_def.patterns:
            matches = sample_str.str.match(pattern)
            pattern_matches += matches.sum()
            
        pattern_score = pattern_matches / len(sample)
        
        # Check for general model number characteristics
        model_characteristics = sample_str.str.match(r'^[A-Z0-9]{2,}[\-]?\d{2,}$|^[A-Z]{1,3}[\-]?\d{3,}$')
        general_score = model_characteristics.sum() / len(sample)
        
        # Final score
        final_score = (pattern_score * 0.3 + general_score * 0.4 + uniqueness_score * 0.3)
        
        return final_score
    
    def _match_category_group(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match Category Group patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        if len(sample) == 0:
            return 0.0
            
        sample_str = sample.astype(str)
        
        # Category Groups typically:
        # 1. Have few unique values (top level categories)
        # 2. Match common AV industry top-level categories
        # 3. Shorter text than specific categories
        
        # Check uniqueness (should be low - many products per category)
        unique_ratio = len(sample.unique()) / len(sample)
        uniqueness_score = 0.0
        if unique_ratio <= 0.1:
            uniqueness_score = 0.8
        elif 0.1 < unique_ratio <= 0.3:
            uniqueness_score = 0.4
            
        # Check for common top-level category names
        top_categories = ['display', 'audio', 'video', 'control', 'infrastructure', 
                        'accessories', 'network', 'conferencing']
        
        # Count matches for each value in the sample
        category_matches = 0
        unique_values = sample_str.unique()
        for value in unique_values:
            value_lower = value.lower()
            if any(cat in value_lower for cat in top_categories):
                category_matches += 1
                
        category_score = category_matches / max(1, len(unique_values))
        
        # Check length (top categories are typically shorter)
        avg_length = sample_str.str.len().mean()
        length_score = 0.0
        if avg_length <= 15:
            length_score = 0.6
        elif 15 < avg_length <= 25:
            length_score = 0.3
            
        # Final score
        final_score = (uniqueness_score * 0.4 + category_score * 0.4 + length_score * 0.2)
        
        return final_score
    
    def _match_category(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match Category patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        if len(sample) == 0:
            return 0.0
            
        sample_str = sample.astype(str)
        
        # Specific Categories typically:
        # 1. Have more unique values than Category Group but still low uniqueness
        # 2. Match common AV industry specific categories
        # 3. Often more detailed than top-level categories
        
        # Check uniqueness (should be low-moderate)
        unique_ratio = len(sample.unique()) / len(sample)
        uniqueness_score = 0.0
        if 0.1 <= unique_ratio <= 0.3:
            uniqueness_score = 0.8
        elif 0.3 < unique_ratio <= 0.5:
            uniqueness_score = 0.4
            
        # Check for common specific category names
        specific_categories = ['projector', 'speaker', 'microphone', 'camera', 'monitor', 
                             'display', 'mount', 'bracket', 'cable', 'adapter', 'converter']
        
        # Count matches for each value in the sample
        category_matches = 0
        unique_values = sample_str.unique()
        for value in unique_values:
            value_lower = value.lower()
            if any(cat in value_lower for cat in specific_categories):
                category_matches += 1
                
        category_score = category_matches / max(1, len(unique_values))
        
        # Check length (specific categories are typically medium length)
        avg_length = sample_str.str.len().mean()
        length_score = 0.0
        if 10 <= avg_length <= 25:
            length_score = 0.6
        elif 5 <= avg_length < 10 or 25 < avg_length <= 40:
            length_score = 0.3
            
        # Final score
        final_score = (uniqueness_score * 0.4 + category_score * 0.4 + length_score * 0.2)
        
        return final_score
    
    def _match_manufacturer(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match Manufacturer patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        if len(sample) == 0:
            return 0.0
            
        sample_str = sample.astype(str)
        
        # Manufacturer fields typically:
        # 1. Have very few unique values
        # 2. Match known AV manufacturers
        # 3. Are consistent (not varying wildly in format)
        
        # Check uniqueness (should be very low - many products per manufacturer)
        unique_ratio = len(sample.unique()) / len(sample)
        uniqueness_score = 0.0
        if unique_ratio <= 0.05:
            uniqueness_score = 0.8
        elif 0.05 < unique_ratio <= 0.2:
            uniqueness_score = 0.4
            
        # Check for known manufacturers
        known_manufacturers = ['sony', 'samsung', 'lg', 'panasonic', 'bose', 'jbl', 'harman', 
                             'crestron', 'extron', 'amx', 'epson', 'christie', 'barco', 
                             'shure', 'sennheiser', 'logitech', 'polycom', 'cisco']
        
        # Count matches for each value in the sample
        mfr_matches = 0
        unique_values = sample_str.unique()
        for value in unique_values:
            value_lower = value.lower()
            if any(mfr.lower() == value_lower or mfr.lower() in value_lower for mfr in known_manufacturers):
                mfr_matches += 1
                
        mfr_score = mfr_matches / max(1, len(unique_values))
        
        # Check consistency (manufacturers usually have consistent formatting)
        length_var = sample_str.str.len().var()
        consistency_score = 0.0
        if length_var <= 10:
            consistency_score = 0.6
        elif 10 < length_var <= 50:
            consistency_score = 0.3
            
        # Final score
        final_score = (uniqueness_score * 0.4 + mfr_score * 0.4 + consistency_score * 0.2)
        
        return final_score
    
    def _match_manufacturer_sku(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match Manufacturer SKU patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        if len(sample) == 0:
            return 0.0
            
        sample_str = sample.astype(str)
        
        # Manufacturer SKUs typically:
        # 1. Are highly unique (like internal SKUs)
        # 2. Often follow manufacturer-specific patterns
        # 3. May differ from internal SKUs in format
        
        # Check uniqueness
        unique_ratio = len(sample.unique()) / len(sample)
        uniqueness_score = 0.0
        if unique_ratio >= 0.9:
            uniqueness_score = 0.7
        elif 0.7 <= unique_ratio < 0.9:
            uniqueness_score = 0.4
            
        # Check for typical manufacturer SKU patterns
        # Many manufacturers use patterns like:
        # - Sony: Usually starts with specific letters followed by numbers/hyphens (e.g., VPL-VW1100ES)
        # - Epson: Often follows pattern like EPSON123456 or V11H123456
        # - Crestron: Often has hyphens and mix of letters/numbers (e.g., DM-MD8X8-CPU3)
        mfr_patterns = [
            r'^[A-Z]{2,4}-[A-Z0-9]{2,}',  # Format like VPL-VW1100
            r'^[A-Z]{1,3}\d{4,}',         # Format like EP6050
            r'^[A-Z]\d{2}[A-Z]\d{3,}'     # Format like V11H123456
        ]
        
        pattern_matches = 0
        for pattern in mfr_patterns:
            matches = sample_str.str.match(pattern)
            pattern_matches += matches.sum()
            
        pattern_score = min(0.8, pattern_matches / len(sample))
        
        # Check that values are not simple numeric IDs (less likely for mfr SKUs)
        simple_numeric = sample_str.str.match(r'^\d+$')
        complexity_score = 0.6 * (1 - (simple_numeric.sum() / len(sample)))
        
        # Final score
        final_score = (uniqueness_score * 0.4 + pattern_score * 0.4 + complexity_score * 0.2)
        
        return final_score
    
    def _match_image_url(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match Image URL patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        if len(sample) == 0:
            return 0.0
            
        sample_str = sample.astype(str)
        
        # Image URLs typically:
        # 1. Start with http:// or https://
        # 2. End with image extensions (.jpg, .png, etc)
        # 3. Contain 'image' or 'img' in the path
        
        # Check for URL patterns
        url_pattern = sample_str.str.match(r'^https?://')
        url_score = url_pattern.sum() / len(sample)
        
        # Check for image extensions
        img_ext_pattern = sample_str.str.match(r'.*\.(jpg|jpeg|png|gif|webp|svg)$', case=False)
        ext_score = img_ext_pattern.sum() / len(sample)
        
        # Check for image-related terms in URL
        img_term_pattern = sample_str.str.contains(r'(image|img|photo|picture)', case=False)
        term_score = img_term_pattern.sum() / len(sample)
        
        # Final score
        final_score = (url_score * 0.5 + ext_score * 0.3 + term_score * 0.2)
        
        return final_score
    
    def _match_document_url(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match Document URL patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        if len(sample) == 0:
            return 0.0
            
        sample_str = sample.astype(str)
        
        # Document URLs typically:
        # 1. Start with http:// or https://
        # 2. End with document extensions (.pdf, .doc, etc)
        # 3. Contain 'doc', 'manual', 'guide', etc. in the path
        
        # Check for URL patterns
        url_pattern = sample_str.str.match(r'^https?://')
        url_score = url_pattern.sum() / len(sample)
        
        # Check for document extensions
        doc_ext_pattern = sample_str.str.match(r'.*\.(pdf|doc|docx|xls|xlsx|ppt|pptx|txt)$', case=False)
        ext_score = doc_ext_pattern.sum() / len(sample)
        
        # Check for document-related terms in URL
        doc_term_pattern = sample_str.str.contains(r'(doc|manual|guide|spec|datasheet)', case=False)
        term_score = doc_term_pattern.sum() / len(sample)
        
        # Final score
        final_score = (url_score * 0.5 + ext_score * 0.3 + term_score * 0.2)
        
        return final_score
    
    def _match_buy_cost(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match Buy Cost patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        return self._match_price_field(sample, field_def, "cost")
    
    def _match_trade_price(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match Trade Price patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        return self._match_price_field(sample, field_def, "trade")
    
    def _match_msrp_gbp(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match MSRP GBP patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        return self._match_price_field(sample, field_def, "msrp", currency="GBP")
    
    def _match_msrp_usd(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match MSRP USD patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        return self._match_price_field(sample, field_def, "msrp", currency="USD")
    
    def _match_msrp_eur(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match MSRP EUR patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        return self._match_price_field(sample, field_def, "msrp", currency="EUR")
    
    def _match_price_field(self, sample: pd.Series, field_def: Any, 
                         price_type: str, currency: str = None) -> float:
        """
        Generic matcher for price fields
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            price_type (str): Type of price (cost, trade, msrp)
            currency (str, optional): Specific currency to look for
            
        Returns:
            float: Confidence score
        """
        if len(sample) == 0:
            return 0.0
            
        # Convert to numeric if possible
        numeric_sample = pd.to_numeric(sample, errors='coerce')
        if numeric_sample.isna().all():
            # Try cleaning strings first
            sample_str = sample.astype(str)
            # Remove currency symbols and commas
            cleaned = sample_str.str.replace(r'[$£€¥,]', '', regex=True)
            numeric_sample = pd.to_numeric(cleaned, errors='coerce')
            
            if numeric_sample.isna().all():
                return 0.0  # Not a numeric field
        
        # Calculate scores based on:
        # 1. Whether values look like prices (numeric with decimals)
        # 2. Value ranges appropriate for price type
        # 3. Currency indicators if specified
        
        # Check decimal precision (prices often have 2 decimal places)
        decimal_count = sum(1 for x in numeric_sample.dropna() 
                          if x != round(x) and round(x * 100) == round(x * 100, 0))
        decimal_score = min(0.7, decimal_count / max(1, len(numeric_sample.dropna())))
        
        # Check value ranges based on price type
        range_score = 0.0
        
        if price_type == "cost":
            # Buy Cost is typically lower than MSRP
            if 0 < numeric_sample.mean() < 5000:
                range_score = 0.5
            elif 0 < numeric_sample.mean() < 10000:
                range_score = 0.3
        elif price_type == "trade":
            # Trade Price is typically between cost and MSRP
            if 0 < numeric_sample.mean() < 7500:
                range_score = 0.5
            elif 0 < numeric_sample.mean() < 15000:
                range_score = 0.3
        elif price_type == "msrp":
            # MSRP is typically higher
            if 0 < numeric_sample.mean() < 10000:
                range_score = 0.5
            elif 0 < numeric_sample.mean() < 20000:
                range_score = 0.3
        
        # Check for currency indicators if specified
        currency_score = 0.5  # Default if currency not specified
        
        if currency:
            sample_str = sample.astype(str)
            
            if currency == "GBP":
                currency_indicators = sample_str.str.contains(r'£|GBP|gbp|pound', regex=True)
                currency_score = currency_indicators.sum() / len(sample) * 0.7
                if currency_score == 0:
                    currency_score = 0.2  # Fallback if no explicit indicators
            elif currency == "USD":
                currency_indicators = sample_str.str.contains(r'\$|USD|usd|dollar', regex=True)
                currency_score = currency_indicators.sum() / len(sample) * 0.7
                if currency_score == 0:
                    currency_score = 0.2
            elif currency == "EUR":
                currency_indicators = sample_str.str.contains(r'€|EUR|eur|euro', regex=True)
                currency_score = currency_indicators.sum() / len(sample) * 0.7
                if currency_score == 0:
                    currency_score = 0.2
        
        # Final score
        if currency:
            final_score = (decimal_score * 0.4 + range_score * 0.3 + currency_score * 0.3)
        else:
            final_score = (decimal_score * 0.5 + range_score * 0.5)
        
        return final_score
    
    def _match_discontinued(self, sample: pd.Series, field_def: Any) -> float:
        """
        Match Discontinued flag patterns
        
        Args:
            sample (pd.Series): Data sample
            field_def: Field definition
            
        Returns:
            float: Confidence score
        """
        if len(sample) == 0:
            return 0.0
            
        sample_str = sample.astype(str).str.lower()
        
        # Discontinued flags typically:
        # 1. Contain yes/no, true/false, or 0/1 values
        # 2. Have very few unique values
        # 3. May contain terms like "discontinued", "active", "available"
        
        # Check for boolean-like values
        bool_pattern = sample_str.isin(['yes', 'no', 'true', 'false', '0', '1', 'y', 'n', 't', 'f'])
        bool_score = bool_pattern.sum() / len(sample)
        
        # Check uniqueness (should be very low, typically 2 values)
        unique_ratio = len(sample_str.unique()) / len(sample)
        uniqueness_score = 0.0
        if unique_ratio <= 0.05:
            uniqueness_score = 0.8
        elif 0.05 < unique_ratio <= 0.2:
            uniqueness_score = 0.4
            
        # Check for status terms
        status_terms = sample_str.str.contains('discontinu|active|available|obsolete|eol|end of life', regex=True)
        term_score = status_terms.sum() / len(sample)
        
        # Final score
        final_score = (bool_score * 0.4 + uniqueness_score * 0.4 + term_score * 0.2)
        
        return final_score