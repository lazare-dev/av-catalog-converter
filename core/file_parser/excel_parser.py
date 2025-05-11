# core/file_parser/excel_parser.py
"""
Excel-specific parsing implementation
"""
import pandas as pd
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from core.file_parser.base_parser import BaseParser
from config.settings import PARSER_CONFIG

class ExcelParser(BaseParser):
    """Parser for Excel files (XLSX, XLS)"""

    def __init__(self, file_path):
        """
        Initialize the Excel parser

        Args:
            file_path (str): Path to the Excel file
        """
        super().__init__(file_path)
        self.config = PARSER_CONFIG['excel']
        self.sheet_name = None
        self.header_row = 0
        self.logger = logging.getLogger(__name__)

    def _get_sheet_names(self):
        """
        Get all sheet names from the Excel file

        Returns:
            list: List of sheet names
        """
        try:
            xl = pd.ExcelFile(self.file_path)
            sheet_names = xl.sheet_names
            self.logger.debug(f"Found sheets: {sheet_names}")
            return sheet_names
        except Exception as e:
            self.logger.error(f"Error reading Excel sheets: {str(e)}")
            return []

    def _select_best_sheet(self):
        """
        Select the most relevant sheet from the workbook

        Returns:
            str: Selected sheet name
        """
        sheet_names = self._get_sheet_names()

        if not sheet_names:
            raise ValueError("No sheets found in Excel file")

        if len(sheet_names) == 1:
            # Only one sheet available
            return sheet_names[0]

        # Look for sheets with catalog-like names
        catalog_keywords = ['product', 'catalog', 'catalogue', 'price', 'inventory', 'stock', 'item']
        for keyword in catalog_keywords:
            for sheet in sheet_names:
                if keyword.lower() in sheet.lower():
                    self.logger.debug(f"Selected sheet '{sheet}' based on keyword '{keyword}'")
                    return sheet

        # If no catalog-like sheet names, look for the sheet with most data
        max_rows = 0
        selected_sheet = sheet_names[0]

        for sheet in sheet_names:
            try:
                # Read a sample to determine row count
                sample = pd.read_excel(self.file_path, sheet_name=sheet, nrows=1)
                sheet_info = pd.read_excel(self.file_path, sheet_name=sheet, header=None, nrows=0)
                num_columns = len(sheet_info.columns)

                # Read last row to determine total rows
                tail_df = pd.read_excel(self.file_path, sheet_name=sheet, skiprows=0, usecols=range(num_columns), nrows=None)
                num_rows = len(tail_df)

                if num_rows > max_rows:
                    max_rows = num_rows
                    selected_sheet = sheet

            except Exception as e:
                self.logger.warning(f"Error analyzing sheet '{sheet}': {str(e)}")
                continue

        self.logger.debug(f"Selected sheet '{selected_sheet}' with {max_rows} rows")
        return selected_sheet

    def _detect_header_row(self, sheet_name):
        """
        Detect which row contains the headers

        Args:
            sheet_name (str): Sheet name to analyze

        Returns:
            int: Row index (0-based) of the header row
        """
        if not self.config['header_detection']:
            return 0

        try:
            # Read first few rows
            sample = pd.read_excel(self.file_path, sheet_name=sheet_name, header=None, nrows=10)

            # Analyze each row as a potential header
            for row_idx in range(min(5, len(sample))):
                row = sample.iloc[row_idx]

                # Check if row has mostly string values (typical for headers)
                str_count = sum(isinstance(val, str) for val in row if not pd.isna(val))
                non_empty = sum(1 for val in row if not pd.isna(val))

                # If most non-empty values are strings, it's likely a header
                if non_empty > 0 and str_count / non_empty > 0.7:
                    self.logger.debug(f"Detected header at row {row_idx}")
                    return row_idx

            # Default to first row
            return 0

        except Exception as e:
            self.logger.warning(f"Error detecting header row: {str(e)}")
            return 0

    def parse(self):
        """
        Parse the Excel file into a pandas DataFrame

        Returns:
            pd.DataFrame: Parsed data
        """
        self.logger.info(f"Parsing Excel file: {self.file_path}")

        # Select the sheet to parse
        if not self.sheet_name:
            self.sheet_name = self._select_best_sheet()

        # Detect header row
        header_row = self._detect_header_row(self.sheet_name)

        try:
            # Get file size for optimization decisions
            import os
            import time
            start_time = time.time()
            file_size = os.path.getsize(self.file_path)
            large_file = file_size > 20 * 1024 * 1024  # 20MB threshold for Excel

            # Optimize for large files
            if large_file:
                self.logger.info(f"Large Excel file detected ({file_size/1024/1024:.1f} MB), using parallel processing")

                # For very large Excel files, use parallel processing with sheet partitioning
                very_large_file = file_size > 100 * 1024 * 1024  # 100MB threshold for very large Excel files

                if very_large_file and self._can_partition_sheet():
                    # Use parallel processing with sheet partitioning
                    from utils.parallel.parallel_processor import ParallelProcessor

                    # Define a function to process a partition of rows
                    def process_partition(partition_info):
                        start_row, end_row = partition_info
                        # Read just this partition of rows
                        return pd.read_excel(
                            self.file_path,
                            sheet_name=self.sheet_name,
                            header=None,  # We'll fix headers later
                            skiprows=start_row,
                            nrows=end_row - start_row,
                            engine='openpyxl',
                            keep_default_na=False
                        )

                    # Get total row count (approximate)
                    total_rows = self._estimate_row_count()

                    # Create partitions
                    partition_size = 10000  # Rows per partition
                    partitions = []
                    for start_row in range(0, total_rows, partition_size):
                        end_row = min(start_row + partition_size, total_rows)
                        if start_row == 0:
                            # First partition includes header
                            partitions.append((header_row, end_row))
                        else:
                            # Skip header for subsequent partitions
                            partitions.append((start_row, end_row))

                    # Process partitions in parallel
                    processor = ParallelProcessor(use_threads=True)  # Threads for I/O-bound Excel reading
                    results = processor._process_chunks(partitions, process_partition)

                    # Combine results
                    if results:
                        # Extract headers from first result
                        headers = results[0].iloc[0] if header_row == 0 else results[0].columns

                        # Process each partition
                        processed_parts = []
                        for i, part in enumerate(results):
                            if i == 0 and header_row == 0:
                                # Skip header row in first partition
                                processed_parts.append(part.iloc[1:])
                            else:
                                processed_parts.append(part)

                        # Combine and set column names
                        df = pd.concat(processed_parts, ignore_index=True)
                        df.columns = headers
                    else:
                        # Fallback if parallel processing failed
                        df = pd.read_excel(
                            self.file_path,
                            sheet_name=self.sheet_name,
                            header=header_row,
                            engine='openpyxl',
                            keep_default_na=False
                        )
                else:
                    # Standard optimized reading for large (but not very large) files
                    df = pd.read_excel(
                        self.file_path,
                        sheet_name=self.sheet_name,
                        header=header_row,
                        engine='openpyxl',  # More memory efficient for large files
                        keep_default_na=False,  # Faster processing
                        na_values=['#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN',
                                  '-NaN', '-nan', '1.#IND', '1.#QNAN', 'N/A', 'NA',
                                  'NULL', 'NaN', 'n/a', 'nan', 'null']
                    )
            else:
                # Standard reading for smaller files
                df = pd.read_excel(
                    self.file_path,
                    sheet_name=self.sheet_name,
                    header=header_row
                )

            # Clean up column names
            df.columns = [str(col).strip() for col in df.columns]

            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')

            # Log performance metrics
            parsing_time = time.time() - start_time
            self.logger.info(f"Successfully parsed Excel sheet '{self.sheet_name}' in {parsing_time:.2f}s",
                            rows=df.shape[0],
                            columns=df.shape[1],
                            file_size_mb=file_size/1024/1024,
                            parsing_speed_rows_per_sec=df.shape[0]/parsing_time if parsing_time > 0 else 0)
            return df

        except Exception as e:
            self.logger.error(f"Error parsing Excel file: {str(e)}")
            raise

    def get_headers(self):
        """
        Extract headers from the Excel file

        Returns:
            list: List of header names
        """
        # Select the sheet to use
        if not self.sheet_name:
            self.sheet_name = self._select_best_sheet()

        # Detect header row
        header_row = self._detect_header_row(self.sheet_name)

        try:
            # Read just the header row
            df = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name,
                header=header_row,
                nrows=0
            )

            headers = list(df.columns)
            self.logger.debug(f"Extracted headers: {headers}")
            return headers

        except Exception as e:
            self.logger.error(f"Error extracting Excel headers: {str(e)}")
            return []

    def _can_partition_sheet(self) -> bool:
        """
        Check if the sheet can be partitioned for parallel processing

        Returns:
            bool: True if the sheet can be partitioned
        """
        try:
            # Check if we can read with skiprows and nrows
            test_df = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name,
                header=None,
                skiprows=0,
                nrows=5
            )
            return len(test_df) > 0
        except Exception as e:
            self.logger.warning(f"Sheet cannot be partitioned: {str(e)}")
            return False

    def _estimate_row_count(self) -> int:
        """
        Estimate the total number of rows in the sheet

        Returns:
            int: Estimated row count
        """
        try:
            # Try to get row count using openpyxl directly
            from openpyxl import load_workbook

            # Load workbook with read-only and data-only options for better performance
            wb = load_workbook(filename=self.file_path, read_only=True, data_only=True)

            # Get the sheet
            if self.sheet_name in wb.sheetnames:
                ws = wb[self.sheet_name]
                # Get dimensions
                if hasattr(ws, 'max_row'):
                    row_count = ws.max_row
                    return row_count

            # If we couldn't get the row count, fall through to the next method
            raise ValueError("Could not determine row count with openpyxl")
        except:
            # Fallback: read a small sample and estimate based on file size
            try:
                sample = pd.read_excel(self.file_path, sheet_name=self.sheet_name, nrows=100)
                file_size = os.path.getsize(self.file_path)

                # Rough estimation based on sample size and file size
                sample_size_bytes = len(sample) * len(sample.columns) * 8  # Assume 8 bytes per cell
                estimated_rows = int((file_size / sample_size_bytes) * len(sample))

                # Cap at a reasonable number
                return min(1000000, max(1000, estimated_rows))
            except:
                # Very conservative fallback
                return 100000

    def get_sample(self, rows=5):
        """
        Get a sample of data for preview

        Args:
            rows (int): Number of rows to sample

        Returns:
            pd.DataFrame: Sample data
        """
        # Select the sheet to use
        if not self.sheet_name:
            self.sheet_name = self._select_best_sheet()

        # Detect header row
        header_row = self._detect_header_row(self.sheet_name)

        try:
            # Read sample rows from the file
            sample_df = pd.read_excel(
                self.file_path,
                sheet_name=self.sheet_name,
                header=header_row,
                nrows=rows
            )

            return sample_df

        except Exception as e:
            self.logger.error(f"Error getting Excel sample: {str(e)}")
            return pd.DataFrame()