# core/file_parser/base_parser.py
"""Base parser interface for all file formats"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import chardet
import re

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

        self.logger.info(f"Initialized parser for {self.file_path}", parser_type=self.__class__.__name__)

    @Logger.get_logger(__name__).logged
    def parse(self) -> pd.DataFrame:
        """
        Parse the file into a pandas DataFrame

        Returns:
            Parsed data as a DataFrame
        """
        raise NotImplementedError("Subclasses must implement parse()")

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
        with self.logger.context(operation="clean_column_names"):
            # Make a copy to avoid modifying the original
            result = df.copy()

            # Clean column names
            rename_dict: Dict[Any, str] = {}
            for col in result.columns:
                # Convert to string if not already
                col_str = str(col)

                # Remove special characters and normalize whitespace
                clean_col = re.sub(r'[^\w\s]', ' ', col_str)
                clean_col = re.sub(r'\s+', ' ', clean_col).strip()

                # If column would be empty after cleaning, use a placeholder
                if not clean_col:
                    clean_col = f"Column_{result.columns.get_loc(col) + 1}"
                    self.logger.debug(f"Empty column name replaced with placeholder",
                                     original=col_str,
                                     placeholder=clean_col)
                elif clean_col != col_str:
                    self.logger.debug(f"Column name cleaned",
                                     original=col_str,
                                     cleaned=clean_col)

                rename_dict[col] = clean_col

            # Apply the renaming
            result.rename(columns=rename_dict, inplace=True)

            # Handle duplicate column names by adding suffixes
            if len(result.columns) != len(set(result.columns)):
                cols = list(result.columns)
                seen: Dict[str, int] = {}
                duplicates: List[str] = []

                for i, col in enumerate(cols):
                    if col in seen:
                        new_col = f"{col}_{seen[col]}"
                        duplicates.append(f"{col} -> {new_col}")
                        cols[i] = new_col
                        seen[col] += 1
                    else:
                        seen[col] = 1

                result.columns = cols
                self.logger.warning(f"Duplicate column names detected and renamed",
                                   duplicates=duplicates)

            self.logger.info(f"Cleaned {len(rename_dict)} column names",
                            column_count=len(result.columns))
            return result

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply common preprocessing to parsed DataFrames

        Args:
            df: Raw parsed DataFrame

        Returns:
            Preprocessed DataFrame
        """
        with self.logger.context(operation="preprocess_dataframe"):
            # Skip if empty
            if df.empty:
                self.logger.warning("Empty DataFrame, skipping preprocessing")
                return df

            # Log initial DataFrame info
            self.logger.info(f"Preprocessing DataFrame",
                            initial_shape=f"{df.shape[0]} rows x {df.shape[1]} columns")

            # Clean column names
            df = self.clean_column_names(df)

            # Remove completely empty rows and columns
            rows_before = df.shape[0]
            cols_before = df.shape[1]

            df = df.dropna(how='all').dropna(axis=1, how='all')

            rows_removed = rows_before - df.shape[0]
            cols_removed = cols_before - df.shape[1]

            if rows_removed > 0 or cols_removed > 0:
                self.logger.info(f"Removed empty rows and columns",
                                rows_removed=rows_removed,
                                cols_removed=cols_removed)

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
                    self.logger.debug(f"Converted column to numeric",
                                     column=col,
                                     valid_ratio=f"{valid_numeric_ratio:.2f}")

            if type_conversions > 0:
                self.logger.info(f"Converted {type_conversions} columns to numeric type")

            self.logger.info(f"Preprocessing complete",
                            final_shape=f"{df.shape[0]} rows x {df.shape[1]} columns")
            return df
