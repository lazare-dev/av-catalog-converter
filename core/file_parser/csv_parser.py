"""CSV file parser implementation with enhanced logging for troubleshooting"""

import pandas as pd
import csv
import os
import time
import traceback
import statistics
from typing import Optional, Union, List, Any, Dict
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
                    lines = []
                    for _ in range(sample_lines):
                        try:
                            line = next(f)
                            lines.append(line)
                        except StopIteration:
                            break

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
                                non_zero_counts = [c for c in counts_per_line if c > 0]
                                if non_zero_counts:
                                    mean = sum(non_zero_counts) / len(non_zero_counts)
                                    # Simple variance calculation
                                    if len(non_zero_counts) > 1:
                                        variance = sum((x - mean) ** 2 for x in non_zero_counts) / len(non_zero_counts)
                                        std = variance ** 0.5
                                    else:
                                        std = 0
                                    consistency[d] = mean / (std + 1)
                                    self.logger.debug(f"Imperfect consistency for delimiter '{d}'",
                                                    mean=mean, std=std, score=consistency[d])
                                else:
                                    consistency[d] = 0
                            except Exception as calc_error:
                                self.logger.warning(f"Error calculating consistency: {str(calc_error)}")
                                consistency[d] = 0

                    self.logger.debug(f"Delimiter consistency scores", consistency=consistency)

                    # Special case for tab-delimited files
                    if '\t' in counts and counts['\t'] > 0:
                        # If file has tabs and they appear consistently, prefer tab as delimiter
                        if consistency.get('\t', 0) > 0:
                            self.delimiter = '\t'
                            self.logger.info(f"Detected tab delimiter",
                                           score=consistency.get('\t', 0),
                                           count=counts['\t'])
                            return '\t'

                    # Get the most consistent delimiter
                    if consistency:
                        best_delimiter = max(consistency.items(), key=lambda x: x[1])[0]
                        best_score = consistency[best_delimiter]
                    else:
                        best_delimiter = ','
                        best_score = 0

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

    def _parse_implementation(self) -> pd.DataFrame:
        """
        Implementation of CSV file parsing with enhanced logging and error handling

        Returns:
            Parsed data as a DataFrame
        """
        with self.logger.context(operation="parse_csv", file_path=str(self.file_path)):
            self.logger.info(f"Parsing CSV file with enhanced logging",
                           parser_type=self.__class__.__name__,
                           file_path=str(self.file_path))

            # Auto-detect parameters if enabled in config
            if self.config.get('encoding_detection', True) and not self.encoding:
                self.logger.debug("Auto-detecting encoding")
                self.detect_encoding()

            if self.config.get('delimiter_detection', True) and not self.delimiter:
                self.logger.debug("Auto-detecting delimiter")
                self.detect_delimiter()

            if self.config.get('header_detection', True) and self.has_header is None:
                self.logger.debug("Auto-detecting header")
                self.detect_header()

            # Set defaults if detection failed
            encoding = self.encoding or 'utf-8'
            delimiter = self.delimiter or ','
            header = 0 if self.has_header else None

            # Log parsing parameters
            self.logger.info("Parsing with detected parameters",
                            encoding=encoding,
                            delimiter=repr(delimiter),
                            has_header=self.has_header,
                            file_path=str(self.file_path))

            try:
                # Get file size for optimization decisions
                import os
                file_size = os.path.getsize(self.file_path)
                large_file = file_size > 50 * 1024 * 1024  # 50MB threshold

                # Log file size information
                self.logger.debug(f"File size information",
                               file_size_bytes=file_size,
                               file_size_mb=file_size/1024/1024,
                               large_file=large_file,
                               threshold_mb=50)

                # Optimize parsing for large files
                if large_file:
                    self.logger.info(f"Large file detected ({file_size/1024/1024:.1f} MB), using parallel processing",
                                   file_path=str(self.file_path),
                                   file_size_mb=file_size/1024/1024)

                    # Use parallel processing for large files
                    from utils.parallel.parallel_processor import ParallelProcessor

                    # Define preprocessing function for each chunk
                    def preprocess_chunk(chunk):
                        # Apply any chunk-specific preprocessing
                        # For now, just return the chunk
                        return chunk

                    # Create parallel processor
                    processor = ParallelProcessor(use_threads=True)  # Use threads for I/O-bound CSV reading

                    # Log parallel processing configuration
                    self.logger.debug(f"Configured parallel processor",
                                    use_threads=True,
                                    processor_type=type(processor).__name__)

                    # Process the file in parallel
                    very_large_file = file_size > 500 * 1024 * 1024  # 500MB threshold for very large files
                    chunk_size = 50000 if very_large_file else 100000  # Smaller chunks for very large files

                    self.logger.debug(f"Parallel processing configuration",
                                    very_large_file=very_large_file,
                                    chunk_size=chunk_size,
                                    file_size_mb=file_size/1024/1024)

                    # Start timing
                    import time
                    start_time = time.time()

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
                        skip_blank_lines=True,
                        # Always ensure values are treated as strings to pass tests
                        dtype=object
                    )

                    # Log parallel processing results
                    processing_time = time.time() - start_time
                    self.logger.info(f"Parallel processing completed",
                                   processing_time_seconds=processing_time,
                                   rows=len(df) if hasattr(df, '__len__') else 'unknown',
                                   columns=len(df.columns) if hasattr(df, 'columns') else 'unknown')
                else:
                    # Standard reading for smaller files
                    self.logger.debug(f"Using standard pandas read_csv for smaller file",
                                    file_size_mb=file_size/1024/1024)

                    # Start timing
                    import time
                    start_time = time.time()

                    df = pd.read_csv(
                        self.file_path,
                        encoding=encoding,
                        delimiter=delimiter,
                        header=header,
                        low_memory=False,
                        on_bad_lines='warn',
                        skipinitialspace=True,
                        skip_blank_lines=True,
                        # Always ensure values are treated as strings to pass tests
                        dtype=object,
                        # Performance optimizations
                        engine='c'  # Use faster C engine
                    )

                    # Log standard processing results
                    processing_time = time.time() - start_time
                    self.logger.info(f"Standard CSV parsing completed",
                                   processing_time_seconds=processing_time,
                                   rows=len(df),
                                   columns=len(df.columns))

                # If no header was detected, create default column names
                if not self.has_header:
                    self.logger.debug("Creating default column names for headerless CSV")
                    df.columns = [f'Column_{i+1}' for i in range(len(df.columns))]
                    self.logger.debug("Created default column names for headerless CSV",
                                    column_count=len(df.columns))

                # Apply common preprocessing
                self.logger.debug("Applying common preprocessing to DataFrame")
                df = self.preprocess_dataframe(df)

                # Log performance metrics
                import time
                parsing_time = time.time() - self.logger.context_data.get('start_time', time.time())
                self.logger.info(f"Successfully parsed CSV in {parsing_time:.2f}s",
                                rows=len(df),
                                columns=len(df.columns),
                                column_names=list(df.columns),
                                file_size_mb=os.path.getsize(self.file_path)/1024/1024 if os.path.exists(self.file_path) else 0,
                                parsing_speed_rows_per_sec=len(df)/parsing_time if parsing_time > 0 else 0,
                                memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024))
                return df

            except Exception as e:
                self.logger.error(f"Error parsing CSV: {str(e)}",
                                 exc_info=True,
                                 stack_info=True,
                                 encoding=encoding,
                                 delimiter=repr(delimiter),
                                 file_path=str(self.file_path),
                                 error_type=type(e).__name__)

                # Try with different encoding if the first attempt failed
                try:
                    self.logger.info("Attempting to parse with utf-8 encoding as fallback",
                                   original_encoding=encoding,
                                   fallback_encoding='utf-8')

                    # Start timing for fallback attempt
                    import time
                    fallback_start_time = time.time()

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

                    # Log fallback success
                    fallback_time = time.time() - fallback_start_time
                    self.logger.info("Successfully parsed CSV with fallback encoding",
                                    rows=len(df),
                                    columns=len(df.columns),
                                    fallback_encoding='utf-8',
                                    processing_time_seconds=fallback_time)
                    return df

                except Exception as inner_e:
                    self.logger.error(f"Failed to parse CSV with alternative encoding: {str(inner_e)}",
                                     exc_info=True,
                                     stack_info=True,
                                     fallback_encoding='utf-8',
                                     original_encoding=encoding,
                                     error_type=type(inner_e).__name__)

                    # Last resort: try to read line by line
                    self.logger.warning("Attempting last resort parsing method: line-by-line",
                                      file_path=str(self.file_path))
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
        Parse CSV line by line as a last resort with enhanced error logging

        Returns:
            Parsed data or empty DataFrame if failed
        """
        with self.logger.context(operation="_parse_line_by_line"):
            self.logger.info("Attempting to parse CSV line by line as last resort",
                           file_path=str(self.file_path),
                           method="line_by_line_fallback")

            # Initialize tracking variables
            rows: List[List[str]] = []
            max_columns = 0
            headers: List[str] = []
            successful_encoding: Optional[str] = None

            # Track detailed statistics for troubleshooting
            stats = {
                'attempted_encodings': [],
                'successful_encoding': None,
                'row_count': 0,
                'max_row_length': 0,
                'min_row_length': float('inf'),
                'empty_rows': 0,
                'inconsistent_rows': 0,
                'file_path': str(self.file_path),
                'delimiter': repr(self.delimiter or ',')
            }

            try:
                # Try different encodings
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    try:
                        self.logger.debug(f"Trying encoding for line-by-line parsing",
                                        encoding=encoding,
                                        file_path=str(self.file_path))

                        stats['attempted_encodings'].append(encoding)

                        with open(self.file_path, 'r', encoding=encoding, errors='replace') as f:
                            reader = csv.reader(f, delimiter=self.delimiter or ',')
                            row_count = 0
                            row_lengths = []

                            for i, row in enumerate(reader):
                                row_count += 1
                                row_lengths.append(len(row))

                                # Track statistics
                                if len(row) == 0:
                                    stats['empty_rows'] += 1

                                stats['max_row_length'] = max(stats['max_row_length'], len(row))
                                if len(row) > 0:
                                    stats['min_row_length'] = min(stats['min_row_length'], len(row))

                                if i == 0 and self.has_header:
                                    headers = row
                                    max_columns = len(row)
                                    self.logger.debug(f"Found header row with {max_columns} columns",
                                                    header_sample=row[:5] if len(row) > 5 else row)
                                else:
                                    rows.append(row)
                                    if len(row) != max_columns and len(row) > 0:
                                        stats['inconsistent_rows'] += 1
                                    max_columns = max(max_columns, len(row))

                                # Log progress for very large files
                                if row_count % 10000 == 0:
                                    self.logger.debug(f"Line-by-line parsing progress",
                                                    rows_processed=row_count,
                                                    encoding=encoding)

                            # If we got here without error, break the loop
                            successful_encoding = encoding
                            stats['successful_encoding'] = encoding
                            stats['row_count'] = row_count

                            # Calculate row length statistics
                            if row_lengths:
                                import statistics
                                try:
                                    stats['mean_row_length'] = statistics.mean(row_lengths)
                                    if len(row_lengths) > 1:
                                        stats['stdev_row_length'] = statistics.stdev(row_lengths)
                                except Exception:
                                    # Fallback if statistics calculation fails
                                    stats['mean_row_length'] = sum(row_lengths) / len(row_lengths)

                            self.logger.info(f"Successfully read {row_count} rows with encoding: {encoding}",
                                           row_count=row_count,
                                           encoding=encoding,
                                           max_columns=max_columns,
                                           inconsistent_rows=stats['inconsistent_rows'],
                                           empty_rows=stats['empty_rows'])
                            break
                    except Exception as e:
                        self.logger.warning(f"Failed to parse with encoding {encoding}",
                                          encoding=encoding,
                                          error=str(e),
                                          error_type=type(e).__name__,
                                          exc_info=True)
                        continue

                if not successful_encoding:
                    self.logger.error("Failed to parse with any encoding",
                                     attempted_encodings=stats['attempted_encodings'],
                                     file_path=str(self.file_path),
                                     exc_info=True,
                                     stack_info=True)
                    return pd.DataFrame()

                # Log detailed statistics before creating DataFrame
                self.logger.debug("Line-by-line parsing statistics", **stats)

                # Create DataFrame
                if self.has_header and headers:
                    # Ensure headers has the right length
                    if len(headers) < max_columns:
                        original_header_len = len(headers)
                        headers.extend([f'Column_{i+1}' for i in range(len(headers), max_columns)])
                        self.logger.debug(f"Extended headers from {original_header_len} to {len(headers)} columns",
                                        original_length=original_header_len,
                                        new_length=len(headers),
                                        max_columns=max_columns)

                    # Start timing DataFrame creation
                    import time
                    df_start_time = time.time()

                    df = pd.DataFrame(rows, columns=headers[:max_columns])

                    df_creation_time = time.time() - df_start_time
                    self.logger.info(f"Created DataFrame with headers",
                                    rows=len(df),
                                    columns=len(df.columns),
                                    creation_time_seconds=df_creation_time,
                                    memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024))
                else:
                    column_names = [f'Column_{i+1}' for i in range(max_columns)]

                    # Start timing DataFrame creation
                    import time
                    df_start_time = time.time()

                    df = pd.DataFrame(rows, columns=column_names)

                    df_creation_time = time.time() - df_start_time
                    self.logger.info(f"Created DataFrame without headers",
                                    rows=len(df),
                                    columns=len(df.columns),
                                    creation_time_seconds=df_creation_time,
                                    memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024))

                # Apply preprocessing
                self.logger.debug("Applying preprocessing to line-by-line parsed DataFrame")
                preprocess_start_time = time.time()
                df = self.preprocess_dataframe(df)
                preprocess_time = time.time() - preprocess_start_time

                self.logger.info(f"Successfully parsed CSV line by line",
                                rows=len(df),
                                columns=len(df.columns),
                                column_names=list(df.columns),
                                encoding=successful_encoding,
                                total_time_seconds=time.time() - self.logger.context_data.get('start_time', time.time()),
                                preprocess_time_seconds=preprocess_time,
                                memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024))
                return df

            except Exception as e:
                # Capture detailed error information
                error_info = {
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'file_path': str(self.file_path),
                    'attempted_encodings': stats['attempted_encodings'],
                    'rows_collected': len(rows),
                    'max_columns': max_columns,
                    'traceback': traceback.format_exc()
                }

                self.logger.error(f"Failed to parse CSV line by line: {str(e)}",
                                exc_info=True,
                                stack_info=True,
                                **error_info)

                self.logger.warning("Returning empty DataFrame as last resort",
                                  file_path=str(self.file_path))
                return pd.DataFrame()  # Return empty DataFrame as last resort
