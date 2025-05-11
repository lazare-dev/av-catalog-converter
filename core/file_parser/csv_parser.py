"""CSV file parser implementation"""

import pandas as pd
import csv
from typing import Optional, Union, List, Any
from pathlib import Path
import numpy as np

from core.file_parser.base_parser import BaseParser
from config.settings import PARSER_CONFIG
from utils.logging.logger import Logger

class CSVParser(BaseParser):
    """Parser for CSV and other delimited text files"""

    def __init__(self, file_path: Union[str, Path]) -> None:
        """
        Initialize the CSV parser

        Args:
            file_path: Path to the CSV file
        """
        super().__init__(file_path)
        self.delimiter: Optional[str] = None
        self.has_header: Optional[bool] = True

        # Get CSV-specific config
        self.config = PARSER_CONFIG.get('csv', {})

        self.logger.info("Initialized CSV parser",
                        file_path=str(self.file_path),
                        config=self.config)

    def detect_delimiter(self, sample_lines: int = 10) -> str:
        """
        Auto-detect the delimiter used in the CSV file

        Args:
            sample_lines: Number of lines to sample

        Returns:
            Detected delimiter character
        """
        with self.logger.context(operation="detect_delimiter", sample_lines=sample_lines):
            encoding = self.encoding or self.detect_encoding()

            # Common delimiters to check
            delimiters = [',', ';', '\t', '|', ':']
            counts = {d: 0 for d in delimiters}

            try:
                with open(self.file_path, 'r', encoding=encoding, errors='replace') as f:
                    # Read sample lines
                    lines = [next(f) for _ in range(sample_lines) if f]

                    self.logger.debug(f"Read {len(lines)} sample lines for delimiter detection")

                    # Count occurrences of each delimiter
                    for line in lines:
                        for d in delimiters:
                            counts[d] += line.count(d)

                    self.logger.debug(f"Delimiter counts", counts=counts)

                    # Find the delimiter with the most consistent count across lines
                    consistency = {}
                    for d in delimiters:
                        if counts[d] == 0:
                            consistency[d] = 0
                            continue

                        # Count delimiters per line
                        counts_per_line = [line.count(d) for line in lines]

                        # Calculate consistency (higher is better)
                        if all(c == counts_per_line[0] and c > 0 for c in counts_per_line):
                            # Perfect consistency
                            consistency[d] = 100 + counts_per_line[0]
                            self.logger.debug(f"Perfect consistency for delimiter '{d}'",
                                             count=counts_per_line[0])
                        else:
                            # Imperfect - use standard deviation as a measure
                            try:
                                import numpy as np
                                non_zero_counts = [c for c in counts_per_line if c > 0]
                                if non_zero_counts:
                                    mean = np.mean(non_zero_counts)
                                    std = np.std(non_zero_counts) if len(non_zero_counts) > 1 else 0
                                    consistency[d] = mean / (std + 1)
                                    self.logger.debug(f"Imperfect consistency for delimiter '{d}'",
                                                    mean=mean, std=std, score=consistency[d])
                                else:
                                    consistency[d] = 0
                            except ImportError:
                                # Fallback if numpy is not available
                                non_zero_counts = [c for c in counts_per_line if c > 0]
                                if non_zero_counts:
                                    mean = sum(non_zero_counts) / len(non_zero_counts)
                                    # Simple variance calculation
                                    variance = sum((x - mean) ** 2 for x in non_zero_counts) / len(non_zero_counts)
                                    std = variance ** 0.5
                                    consistency[d] = mean / (std + 1)
                                    self.logger.debug(f"Imperfect consistency for delimiter '{d}' (fallback)",
                                                    mean=mean, std=std, score=consistency[d])
                                else:
                                    consistency[d] = 0

                    self.logger.debug(f"Delimiter consistency scores", consistency=consistency)

                    # Get the most consistent delimiter
                    best_delimiter = max(consistency.items(), key=lambda x: x[1])[0]
                    best_score = consistency[best_delimiter]

                    # If the best delimiter has very low consistency, default to comma
                    if best_score < 1:
                        self.logger.warning("Low delimiter consistency, defaulting to comma",
                                          best_delimiter=best_delimiter,
                                          best_score=best_score)
                        best_delimiter = ','

                    self.delimiter = best_delimiter
                    self.logger.info(f"Detected delimiter: '{best_delimiter}'",
                                    score=best_score,
                                    count=counts[best_delimiter])
                    return best_delimiter

            except Exception as e:
                self.logger.error(f"Error detecting delimiter: {str(e)}",
                                 exc_info=True,
                                 file_path=str(self.file_path))
                self.delimiter = ','
                self.logger.info("Defaulting to comma delimiter due to error")
                return ','

    def detect_header(self) -> bool:
        """
        Detect if the CSV file has a header row

        Returns:
            True if header detected, False otherwise
        """
        with self.logger.context(operation="detect_header"):
            encoding = self.encoding or self.detect_encoding()
            delimiter = self.delimiter or self.detect_delimiter()

            try:
                with open(self.file_path, 'r', encoding=encoding, errors='replace') as f:
                    # Read a few lines
                    sample = [next(f) for _ in range(3) if f]

                    if not sample:
                        self.logger.warning("Empty file or couldn't read sample lines, assuming header exists")
                        self.has_header = True
                        return True

                    self.logger.debug(f"Read {len(sample)} sample lines for header detection")

                    # Parse the first few lines
                    reader = csv.reader(sample, delimiter=delimiter)
                    rows = list(reader)

                    if len(rows) < 2:
                        self.logger.warning("Not enough rows to detect header, assuming header exists")
                        self.has_header = True
                        return True

                    first_row = rows[0]
                    second_row = rows[1]

                    # Check if first row looks like a header
                    # 1. Headers often contain string data while data rows have mixed types
                    first_row_all_strings = all(not cell.replace('.', '').isdigit() for cell in first_row if cell)
                    second_row_has_numbers = any(cell.replace('.', '').isdigit() for cell in second_row if cell)

                    # 2. Headers often have different formatting (e.g., capitalization, underscores)
                    first_row_formatting = any('_' in cell or cell.isupper() or cell.istitle() for cell in first_row if cell)

                    # 3. Headers are often shorter than data
                    avg_first_len = sum(len(cell) for cell in first_row) / max(1, len(first_row))
                    avg_second_len = sum(len(cell) for cell in second_row) / max(1, len(second_row))

                    # Log the evidence
                    self.logger.debug("Header detection evidence",
                                     first_row_all_strings=first_row_all_strings,
                                     second_row_has_numbers=second_row_has_numbers,
                                     first_row_formatting=first_row_formatting,
                                     avg_first_len=avg_first_len,
                                     avg_second_len=avg_second_len)

                    # Combine evidence
                    evidence = []
                    if first_row_all_strings and second_row_has_numbers:
                        evidence.append("string headers with numeric data")
                    if first_row_formatting:
                        evidence.append("header formatting (capitalization, underscores)")
                    if avg_first_len < avg_second_len * 0.7:
                        evidence.append("headers shorter than data")

                    has_header = (first_row_all_strings and second_row_has_numbers) or \
                                 first_row_formatting or \
                                 (avg_first_len < avg_second_len * 0.7)

                    self.has_header = has_header

                    if has_header:
                        self.logger.info("Header row detected",
                                        evidence=evidence,
                                        header_sample=first_row[:5] if len(first_row) > 5 else first_row)
                    else:
                        self.logger.info("No header row detected",
                                        first_row_sample=first_row[:5] if len(first_row) > 5 else first_row)

                    return has_header

            except Exception as e:
                self.logger.error(f"Error detecting header: {str(e)}",
                                 exc_info=True,
                                 file_path=str(self.file_path))
                self.has_header = True
                self.logger.info("Defaulting to assume header exists due to error")
                return True

    @Logger.get_logger(__name__).logged
    def parse(self) -> pd.DataFrame:
        """
        Parse the CSV file into a pandas DataFrame

        Returns:
            Parsed data as a DataFrame
        """
        with self.logger.context(operation="parse", file_path=str(self.file_path)):
            self.logger.info(f"Parsing CSV file")

            # Auto-detect parameters if enabled in config
            if self.config.get('encoding_detection', True) and not self.encoding:
                self.detect_encoding()

            if self.config.get('delimiter_detection', True) and not self.delimiter:
                self.detect_delimiter()

            if self.config.get('header_detection', True) and self.has_header is None:
                self.detect_header()

            # Set defaults if detection failed
            encoding = self.encoding or 'utf-8'
            delimiter = self.delimiter or ','
            header = 0 if self.has_header else None

            # Log parsing parameters
            self.logger.info("Parsing with detected parameters",
                            encoding=encoding,
                            delimiter=repr(delimiter),
                            has_header=self.has_header)

            try:
                # Get file size for optimization decisions
                import os
                file_size = os.path.getsize(self.file_path)
                large_file = file_size > 50 * 1024 * 1024  # 50MB threshold

                # Optimize parsing for large files
                if large_file:
                    self.logger.info(f"Large file detected ({file_size/1024/1024:.1f} MB), using parallel processing")

                    # Use parallel processing for large files
                    from utils.parallel.parallel_processor import ParallelProcessor

                    # Define preprocessing function for each chunk
                    def preprocess_chunk(chunk):
                        # Apply any chunk-specific preprocessing
                        # For now, just return the chunk
                        return chunk

                    # Create parallel processor
                    processor = ParallelProcessor(use_threads=True)  # Use threads for I/O-bound CSV reading

                    # Process the file in parallel
                    very_large_file = file_size > 500 * 1024 * 1024  # 500MB threshold for very large files
                    chunk_size = 50000 if very_large_file else 100000  # Smaller chunks for very large files

                    df = processor.process_file(
                        self.file_path,
                        process_func=preprocess_chunk,
                        chunk_size=chunk_size,
                        encoding=encoding,
                        delimiter=delimiter,
                        header=header,
                        low_memory=True,
                        on_bad_lines='warn',
                        skipinitialspace=True,
                        skip_blank_lines=True
                    )
                else:
                    # Standard reading for smaller files
                    df = pd.read_csv(
                        self.file_path,
                        encoding=encoding,
                        delimiter=delimiter,
                        header=header,
                        low_memory=False,
                        on_bad_lines='warn',
                        skipinitialspace=True,
                        skip_blank_lines=True,
                        # Performance optimizations
                        engine='c',  # Use faster C engine
                        dtype_backend='numpy_nullable'  # More efficient memory usage
                    )

                # If no header was detected, create default column names
                if not self.has_header:
                    df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]
                    self.logger.debug("Created default column names for headerless CSV")

                # Apply common preprocessing
                df = self.preprocess_dataframe(df)

                # Log performance metrics
                import time
                parsing_time = time.time() - self.logger.context_data.get('start_time', time.time())
                self.logger.info(f"Successfully parsed CSV in {parsing_time:.2f}s",
                                rows=len(df),
                                columns=len(df.columns),
                                file_size_mb=os.path.getsize(self.file_path)/1024/1024 if os.path.exists(self.file_path) else 0,
                                parsing_speed_rows_per_sec=len(df)/parsing_time if parsing_time > 0 else 0)
                return df

            except Exception as e:
                self.logger.error(f"Error parsing CSV: {str(e)}",
                                 exc_info=True,
                                 encoding=encoding,
                                 delimiter=repr(delimiter))

                # Try with different encoding if the first attempt failed
                try:
                    self.logger.info("Attempting to parse with utf-8 encoding as fallback")
                    df = pd.read_csv(
                        self.file_path,
                        encoding='utf-8',
                        delimiter=delimiter,
                        header=header,
                        low_memory=False,
                        on_bad_lines='warn'
                    )

                    if not self.has_header:
                        df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]

                    df = self.preprocess_dataframe(df)

                    self.logger.info("Successfully parsed CSV with fallback encoding",
                                    rows=len(df),
                                    columns=len(df.columns))
                    return df

                except Exception as inner_e:
                    self.logger.error(f"Failed to parse CSV with alternative encoding: {str(inner_e)}",
                                     exc_info=True)

                    # Last resort: try to read line by line
                    return self._parse_line_by_line()

    def get_headers(self) -> List[str]:
        """
        Get the headers from the CSV file

        Returns:
            List[str]: List of header names
        """
        with self.logger.context(operation="get_headers"):
            try:
                # Auto-detect parameters if needed
                encoding = self.encoding or self.detect_encoding()
                delimiter = self.delimiter or self.detect_delimiter()
                has_header = self.has_header if self.has_header is not None else self.detect_header()

                # Read just the first row
                df = pd.read_csv(
                    self.file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    header=0 if has_header else None,
                    nrows=1
                )

                # Return column names
                headers = list(df.columns)
                self.logger.info(f"Retrieved {len(headers)} headers from CSV file")
                return headers

            except Exception as e:
                self.logger.error(f"Error getting headers: {str(e)}", exc_info=True)
                return []

    def _parse_line_by_line(self) -> pd.DataFrame:
        """
        Parse CSV line by line as a last resort

        Returns:
            Parsed data or empty DataFrame if failed
        """
        with self.logger.context(operation="_parse_line_by_line"):
            self.logger.info("Attempting to parse CSV line by line as last resort")
            rows: List[List[str]] = []
            max_columns = 0
            headers: List[str] = []
            successful_encoding: Optional[str] = None

            try:
                # Try different encodings
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        self.logger.debug(f"Trying encoding: {encoding}")
                        with open(self.file_path, 'r', encoding=encoding, errors='replace') as f:
                            reader = csv.reader(f, delimiter=self.delimiter or ',')
                            row_count = 0

                            for i, row in enumerate(reader):
                                row_count += 1
                                if i == 0 and self.has_header:
                                    headers = row
                                    max_columns = len(row)
                                    self.logger.debug(f"Found header row with {max_columns} columns")
                                else:
                                    rows.append(row)
                                    max_columns = max(max_columns, len(row))

                            # If we got here without error, break the loop
                            successful_encoding = encoding
                            self.logger.info(f"Successfully read {row_count} rows with encoding: {encoding}")
                            break
                    except Exception as e:
                        self.logger.warning(f"Failed to parse with encoding {encoding}: {str(e)}")
                        continue

                if not successful_encoding:
                    self.logger.error("Failed to parse with any encoding")
                    return pd.DataFrame()

                # Create DataFrame
                if self.has_header and headers:
                    # Ensure headers has the right length
                    if len(headers) < max_columns:
                        original_header_len = len(headers)
                        headers.extend([f'Column_{i+1}' for i in range(len(headers), max_columns)])
                        self.logger.debug(f"Extended headers from {original_header_len} to {len(headers)} columns")

                    df = pd.DataFrame(rows, columns=headers[:max_columns])
                    self.logger.info(f"Created DataFrame with headers",
                                    rows=len(df),
                                    columns=len(df.columns))
                else:
                    column_names = [f'Column_{i+1}' for i in range(max_columns)]
                    df = pd.DataFrame(rows, columns=column_names)
                    self.logger.info(f"Created DataFrame without headers",
                                    rows=len(df),
                                    columns=len(df.columns))

                # Apply preprocessing
                df = self.preprocess_dataframe(df)

                self.logger.info(f"Successfully parsed CSV line by line",
                                rows=len(df),
                                columns=len(df.columns),
                                encoding=successful_encoding)
                return df

            except Exception as e:
                self.logger.error(f"Failed to parse CSV line by line: {str(e)}", exc_info=True)
                self.logger.warning("Returning empty DataFrame as last resort")
                return pd.DataFrame()  # Return empty DataFrame as last resort
