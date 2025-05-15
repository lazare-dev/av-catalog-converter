# core/file_parser/base_parser.py
"""Base parser interface for all file formats"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import chardet
import re
import traceback

from utils.logging.logger import Logger

class BaseParser:
    """Base class for all file parsers"""

    def __init__(self, file_path: Union[str, Path]) -> None:
        """
        Initialize the parser

        Args:
            file_path: Path to the input file
        """
        self.file_path = Path(file_path)
        self.logger = Logger.get_logger(__name__)
        self.encoding: Optional[str] = None
        self.detected_headers: Optional[List[str]] = None
        self.data_boundaries: Optional[Dict[str, int]] = None

        # Gather detailed file information for troubleshooting
        file_info = {}
        if self.file_path.exists():
            try:
                file_stat = self.file_path.stat()
                file_info = {
                    'absolute_path': str(self.file_path.absolute()),
                    'size_bytes': file_stat.st_size,
                    'size_mb': file_stat.st_size / (1024 * 1024),
                    'modified_time': file_stat.st_mtime,
                    'access_time': file_stat.st_atime,
                    'extension': self.file_path.suffix.lower(),
                    'parent_dir': str(self.file_path.parent),
                    'exists': True,
                    'is_file': self.file_path.is_file(),
                    'is_symlink': self.file_path.is_symlink(),
                    'readable': os.access(self.file_path, os.R_OK)
                }
            except Exception as e:
                file_info = {
                    'error': str(e),
                    'absolute_path': str(self.file_path.absolute()),
                    'exists': self.file_path.exists()
                }
        else:
            file_info = {
                'error': 'File does not exist',
                'absolute_path': str(self.file_path.absolute()),
                'exists': False
            }

        self.logger.info(f"Initialized parser for {self.file_path}",
                        parser_type=self.__class__.__name__,
                        file_info=file_info)

    def parse(self) -> pd.DataFrame:
        """
        Parse the file into a pandas DataFrame with enhanced error logging

        Returns:
            Parsed data as a DataFrame
        """
        logger = self.logger
        logger.debug(f"Starting to parse file",
                    file_path=str(self.file_path),
                    parser_type=self.__class__.__name__)

        try:
            # This will be implemented by subclasses
            start_time = pd.Timestamp.now()
            logger.debug(f"Parsing started at {start_time}")

            # The actual implementation will be in subclasses
            result = self._parse_implementation()

            end_time = pd.Timestamp.now()
            duration = (end_time - start_time).total_seconds()

            # Log detailed information about the parsing result
            if isinstance(result, pd.DataFrame):
                logger.info(f"Successfully parsed file",
                           file_path=str(self.file_path),
                           parser_type=self.__class__.__name__,
                           rows=len(result),
                           columns=len(result.columns),
                           column_names=list(result.columns),
                           duration_seconds=duration,
                           memory_usage_bytes=result.memory_usage(deep=True).sum(),
                           memory_usage_mb=result.memory_usage(deep=True).sum() / (1024 * 1024))
            else:
                logger.warning(f"Parse result is not a DataFrame",
                              file_path=str(self.file_path),
                              result_type=type(result).__name__)

            return result
        except Exception as e:
            # Capture detailed error information
            error_info = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'file_path': str(self.file_path),
                'parser_type': self.__class__.__name__,
                'traceback': traceback.format_exc()
            }

            # Add file information if available
            if hasattr(self, 'encoding') and self.encoding:
                error_info['encoding'] = self.encoding

            logger.error(f"Error parsing file: {str(e)}",
                        exc_info=True,
                        stack_info=True,
                        **error_info)

            # Re-raise the exception
            raise

    def _parse_implementation(self) -> pd.DataFrame:
        """
        Actual implementation of the parsing logic.
        This method should be overridden by subclasses.

        Returns:
            Parsed data as a DataFrame
        """
        raise NotImplementedError("Subclasses must implement _parse_implementation()")

    def get_sample(self, rows: int = 5) -> pd.DataFrame:
        """
        Get a sample of the parsed data

        Args:
            rows: Number of rows to sample

        Returns:
            Sample data as a DataFrame
        """
        with self.logger.context(operation="get_sample", rows=rows):
            self.logger.info(f"Getting sample of {rows} rows")
            data = self.parse()
            sample_size = min(rows, len(data))
            sample = data.head(sample_size)
            self.logger.info(f"Sample retrieved",
                            sample_size=sample_size,
                            columns=len(sample.columns))
            return sample

    def detect_encoding(self, sample_size: int = 10000) -> str:
        """
        Detect file encoding

        Args:
            sample_size: Number of bytes to sample

        Returns:
            str: Detected encoding
        """
        with self.logger.context(operation="detect_encoding", file_path=str(self.file_path)):
            with open(self.file_path, 'rb') as f:
                raw_data = f.read(sample_size)
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']

                self.logger.debug(
                    f"Detected encoding: {encoding}",
                    encoding=encoding,
                    confidence=f"{confidence:.2f}"
                )

                # Default to utf-8 if detection failed or low confidence
                if not encoding or confidence < 0.7:
                    encoding = 'utf-8'
                    self.logger.warning(
                        f"Low confidence encoding detection, defaulting to {encoding}",
                        original_encoding=encoding,
                        confidence=f"{confidence:.2f}",
                        default_encoding="utf-8"
                    )

                self.encoding = encoding
                return encoding

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize column names

        Args:
            df: DataFrame with original column names

        Returns:
            DataFrame with cleaned column names
        """
        # Make a copy to avoid modifying the original
        result = df.copy()

        # Get the original column list - this preserves duplicates
        original_columns = list(df.columns)

        # Clean column names
        rename_dict: Dict[Any, str] = {}
        for col in original_columns:
            # Convert to string if not already
            col_str = str(col)

            # Remove special characters and normalize whitespace
            clean_col = re.sub(r'[^\w\s]', ' ', col_str)
            clean_col = re.sub(r'\s+', ' ', clean_col).strip()

            # If column would be empty after cleaning, use a placeholder
            if not clean_col:
                clean_col = f"Column_{original_columns.index(col) + 1}"
                self.logger.debug(f"Empty column name replaced with placeholder: {col_str} -> {clean_col}")
            elif clean_col != col_str:
                self.logger.debug(f"Column name cleaned: {col_str} -> {clean_col}")

            rename_dict[col] = clean_col

        # Create a new DataFrame with the cleaned column names
        # This approach preserves duplicate columns
        new_columns = [rename_dict.get(col, col) for col in original_columns]

        # For testing purposes, print the columns to debug
        self.logger.debug(f"Original columns: {list(original_columns)}")
        self.logger.debug(f"New columns before setting: {new_columns}")

        # Create a new DataFrame with the same data but new column names
        # This is necessary because pandas silently drops duplicate column names
        new_df = pd.DataFrame(result.values, columns=new_columns)
        result = new_df

        # Handle duplicate column names by adding suffixes
        if len(set(new_columns)) < len(new_columns):
            # Create a dictionary to track seen columns and their counts
            seen = {}
            # Track renamed columns for logging
            duplicates = []

            # Create a new list for the renamed columns
            final_cols = []

            for col in new_columns:
                if col in seen:
                    # This is a duplicate, add a suffix with the count
                    seen[col] += 1
                    new_col = f"{col}_{seen[col]}"
                    duplicates.append(f"{col} -> {new_col}")
                    final_cols.append(new_col)
                else:
                    # First time seeing this column
                    seen[col] = 0
                    final_cols.append(col)

            # Set the new column names
            result.columns = final_cols

            self.logger.warning(f"Duplicate column names detected and renamed: {duplicates}")

        self.logger.info(f"Cleaned {len(rename_dict)} column names, final column count: {len(result.columns)}")
        return result

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply common preprocessing to parsed DataFrames

        Args:
            df: Raw parsed DataFrame

        Returns:
            Preprocessed DataFrame
        """
        # Skip if empty
        if df.empty:
            self.logger.warning("Empty DataFrame, skipping preprocessing")
            return df

        # Log initial DataFrame info
        self.logger.info(f"Preprocessing DataFrame with {df.shape[0]} rows x {df.shape[1]} columns")

        # Clean column names
        df = self.clean_column_names(df)

        # Remove completely empty rows and columns
        rows_before = df.shape[0]
        cols_before = df.shape[1]

        df = df.dropna(how='all').dropna(axis=1, how='all')

        rows_removed = rows_before - df.shape[0]
        cols_removed = cols_before - df.shape[1]

        if rows_removed > 0 or cols_removed > 0:
            self.logger.info(f"Removed {rows_removed} empty rows and {cols_removed} empty columns")

        # Convert 'None', 'NULL', 'N/A' strings to actual NaN
        na_values = ['none', 'null', 'n/a', 'na', '#n/a', '#na', '#value!', '#ref!']
        na_replacements = 0

        for col in df.columns:
            if df[col].dtype == 'object':
                # Count NaN values before replacement
                na_before = df[col].isna().sum()

                # Replace NA strings
                df[col] = df[col].apply(
                    lambda x: pd.NA if isinstance(x, str) and x.lower().strip() in na_values else x
                )

                # Count NaN values after replacement
                na_after = df[col].isna().sum()
                replacements = na_after - na_before

                if replacements > 0:
                    na_replacements += replacements

        if na_replacements > 0:
            self.logger.info(f"Replaced {na_replacements} NA string values with NaN")

        # Try to infer better data types
        type_conversions = 0

        for col in df.columns:
            # Skip columns that are already non-object type
            if df[col].dtype != 'object':
                continue

            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            valid_numeric_ratio = numeric_series.notna().sum() / df[col].count() if df[col].count() > 0 else 0

            if valid_numeric_ratio > 0.5:
                # If more than 50% of values can be converted to numeric, do it
                df[col] = numeric_series
                type_conversions += 1
                self.logger.debug(f"Converted column {col} to numeric (valid ratio: {valid_numeric_ratio:.2f})")

        if type_conversions > 0:
            self.logger.info(f"Converted {type_conversions} columns to numeric type")

        self.logger.info(f"Preprocessing complete, final shape: {df.shape[0]} rows x {df.shape[1]} columns")
        return df
