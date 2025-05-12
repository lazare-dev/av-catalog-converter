"""Value normalization service"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.schema import FIELD_ORDER, SCHEMA_DICT, NUMERIC_FIELDS
from services.normalization.text_normalizer import TextNormalizer
from services.normalization.price_normalizer import PriceNormalizer
from services.normalization.id_normalizer import IDNormalizer
from services.normalization.unit_normalizer import UnitNormalizer
from utils.helpers.profiling_helpers import timeit

class ValueNormalizer:
    """Service for normalizing values across the standardized schema"""

    def __init__(self, max_workers=None):
        """
        Initialize the value normalizer

        Args:
            max_workers (int, optional): Maximum number of worker threads
        """
        self.logger = logging.getLogger(__name__)
        self.text_normalizer = TextNormalizer()
        self.price_normalizer = PriceNormalizer()
        self.id_normalizer = IDNormalizer()
        self.unit_normalizer = UnitNormalizer()
        self.max_workers = max_workers

    @timeit
    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize values according to field types

        Args:
            data (pd.DataFrame): Input data with mapped fields

        Returns:
            pd.DataFrame: Data with normalized values
        """
        self.logger.info("Starting value normalization")

        # Make a copy to avoid modifying the original
        result_df = data.copy()

        # Group fields by type for batch processing
        text_fields = []
        price_fields = []
        id_fields = []
        unit_fields = []
        other_fields = []

        for field in FIELD_ORDER:
            if field not in result_df.columns:
                continue

            if field in ["Short Description", "Long Description", "Document Name"]:
                text_fields.append(field)
            elif field in ["Buy Cost", "Trade Price", "MSRP GBP", "MSRP USD", "MSRP EUR"]:
                price_fields.append(field)
            elif field in ["SKU", "Model", "Manufacturer SKU"]:
                id_fields.append(field)
            elif field in ["Unit Of Measure"]:
                unit_fields.append(field)
            else:
                other_fields.append(field)

        # Use parallel processing for large datasets
        if len(result_df) > 1000 and self.max_workers != 1:
            self.logger.info(f"Using parallel processing for {len(result_df)} rows")
            result_df = self._normalize_parallel(result_df, text_fields, price_fields, id_fields, unit_fields)
        else:
            # Process each field group
            if text_fields:
                self.logger.debug(f"Normalizing text fields: {text_fields}")
                for field in text_fields:
                    result_df[field] = self.text_normalizer.normalize_text(result_df[field])

            if price_fields:
                self.logger.debug(f"Normalizing price fields: {price_fields}")
                for field in price_fields:
                    result_df[field] = self.price_normalizer.normalize_price(result_df[field], field)

            if id_fields:
                self.logger.debug(f"Normalizing ID fields: {id_fields}")
                for field in id_fields:
                    result_df[field] = self.id_normalizer.normalize_id(result_df[field], field)

            if unit_fields:
                self.logger.debug(f"Normalizing unit fields: {unit_fields}")
                for field in unit_fields:
                    result_df[field] = self.unit_normalizer.normalize_unit(result_df[field])

        # Handle boolean fields
        if "Discontinued" in result_df.columns:
            self.logger.debug("Normalizing Discontinued field")
            result_df["Discontinued"] = self._normalize_boolean(result_df["Discontinued"])

        self.logger.info("Value normalization complete")
        return result_df

    def _normalize_parallel(self, df: pd.DataFrame, text_fields: List[str],
                           price_fields: List[str], id_fields: List[str],
                           unit_fields: List[str]) -> pd.DataFrame:
        """
        Normalize values using parallel processing

        Args:
            df (pd.DataFrame): Input DataFrame
            text_fields (List[str]): Text fields to normalize
            price_fields (List[str]): Price fields to normalize
            id_fields (List[str]): ID fields to normalize
            unit_fields (List[str]): Unit fields to normalize

        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        result = df.copy()

        # Split DataFrame into chunks
        from core.chunking.row_chunker import RowChunker
        chunker = RowChunker(chunk_size=1000)
        chunks = chunker.split(df)

        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            futures = []

            for i, chunk in enumerate(chunks):
                future = executor.submit(
                    self._process_chunk,
                    chunk,
                    text_fields,
                    price_fields,
                    id_fields,
                    unit_fields,
                    i
                )
                futures.append(future)

            # Collect results
            normalized_chunks = []
            for future in as_completed(futures):
                try:
                    chunk_result = future.result()
                    normalized_chunks.append(chunk_result)
                except Exception as e:
                    self.logger.error(f"Error in parallel normalization: {str(e)}")

            # Combine chunks
            if normalized_chunks:
                result = pd.concat(normalized_chunks, ignore_index=True)

        return result

    def _process_chunk(self, chunk: pd.DataFrame, text_fields: List[str],
                      price_fields: List[str], id_fields: List[str],
                      unit_fields: List[str], chunk_id: int) -> pd.DataFrame:
        """
        Process a single chunk of data

        Args:
            chunk (pd.DataFrame): Data chunk
            text_fields (List[str]): Text fields to normalize
            price_fields (List[str]): Price fields to normalize
            id_fields (List[str]): ID fields to normalize
            unit_fields (List[str]): Unit fields to normalize
            chunk_id (int): Chunk identifier

        Returns:
            pd.DataFrame: Normalized chunk
        """
        self.logger.debug(f"Processing chunk {chunk_id} with {len(chunk)} rows")
        result = chunk.copy()

        # Process each field group
        for field in text_fields:
            if field in result.columns:
                result[field] = self.text_normalizer.normalize_text(result[field])

        for field in price_fields:
            if field in result.columns:
                result[field] = self.price_normalizer.normalize_price(result[field], field)

        for field in id_fields:
            if field in result.columns:
                result[field] = self.id_normalizer.normalize_id(result[field], field)

        for field in unit_fields:
            if field in result.columns:
                result[field] = self.unit_normalizer.normalize_unit(result[field])

        return result

    def _normalize_boolean(self, series: pd.Series) -> pd.Series:
        """
        Normalize boolean values to Yes/No

        Args:
            series (pd.Series): Input series

        Returns:
            pd.Series: Normalized series
        """
        # Map various values to Yes/No
        yes_values = ['yes', 'y', 'true', '1', 't', 'discontinued', 'obsolete', 'eol']
        result = series.astype(str).str.lower()

        return result.apply(
            lambda x: "Yes" if x in yes_values else "No"
        )

    def _normalize_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize text fields in the DataFrame

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with normalized text fields
        """
        result = df.copy()
        text_fields = ["Short Description", "Long Description", "Document Name"]

        for field in text_fields:
            if field in result.columns:
                result[field] = self.text_normalizer.normalize_text(result[field])

        return result

    def _normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize price fields in the DataFrame

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with normalized price fields
        """
        result = df.copy()
        price_fields = ["Buy Cost", "Trade Price", "MSRP GBP", "MSRP USD", "MSRP EUR"]

        for field in price_fields:
            if field in result.columns:
                result[field] = self.price_normalizer.normalize_price(result[field], field)

        return result

    def _normalize_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize ID fields in the DataFrame

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with normalized ID fields
        """
        result = df.copy()
        id_fields = ["SKU", "Model", "Manufacturer SKU"]

        for field in id_fields:
            if field in result.columns:
                result[field] = self.id_normalizer.normalize_id(result[field], field)

        return result
