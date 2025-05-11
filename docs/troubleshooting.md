# AV Catalog Converter Troubleshooting Guide

This guide provides solutions to common issues you might encounter when using the AV Catalog Converter.

## Table of Contents

- [Installation Issues](#installation-issues)
- [File Parsing Issues](#file-parsing-issues)
- [Field Mapping Issues](#field-mapping-issues)
- [Performance Issues](#performance-issues)
- [API Issues](#api-issues)
- [Frontend Issues](#frontend-issues)
- [Docker Issues](#docker-issues)
- [LLM Integration Issues](#llm-integration-issues)
- [Common Error Messages](#common-error-messages)

## Installation Issues

### Python Dependencies

**Issue**: Error installing dependencies from requirements.txt

**Solution**:
1. Ensure you're using Python 3.8 or higher:
   ```bash
   python --version
   ```

2. Try updating pip:
   ```bash
   pip install --upgrade pip
   ```

3. Install dependencies one by one to identify problematic packages:
   ```bash
   pip install pandas
   pip install numpy==1.24.3
   # Continue with other packages
   ```

### Tesseract OCR Installation

**Issue**: PDF parsing fails with "Tesseract not found" error

**Solution**:
1. Install Tesseract OCR:
   - **Windows**: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - **Linux**: `sudo apt-get install tesseract-ocr`
   - **macOS**: `brew install tesseract`

2. Ensure Tesseract is in your PATH:
   ```bash
   # Check if Tesseract is installed
   tesseract --version
   ```

3. Set the Tesseract path explicitly in config/settings.yaml:
   ```yaml
   ocr:
     tesseract_path: /path/to/tesseract
   ```

### Node.js and npm Issues

**Issue**: Frontend dependencies fail to install

**Solution**:
1. Ensure you have Node.js 14+ and npm installed:
   ```bash
   node --version
   npm --version
   ```

2. Clear npm cache:
   ```bash
   npm cache clean --force
   ```

3. Try using a specific npm registry:
   ```bash
   npm config set registry https://registry.npmjs.org/
   npm install
   ```

## File Parsing Issues

### CSV Parsing Issues

**Issue**: CSV files with non-standard delimiters or encodings fail to parse

**Solution**:
1. Check the file encoding and delimiter:
   ```bash
   file -i your_file.csv
   ```

2. Specify the delimiter and encoding explicitly:
   ```bash
   python app.py --input your_file.csv --delimiter ";" --encoding "latin-1"
   ```

3. Pre-process the file to convert to standard CSV:
   ```bash
   iconv -f original_encoding -t utf-8 original_file.csv > converted_file.csv
   ```

### Excel Parsing Issues

**Issue**: Excel files with complex formatting or macros fail to parse

**Solution**:
1. Save the Excel file as a simple XLSX without macros or complex formatting

2. Try using the `--sheet` option to specify a particular sheet:
   ```bash
   python app.py --input your_file.xlsx --sheet "Sheet1"
   ```

3. For very large Excel files, convert to CSV first:
   ```bash
   python -c "import pandas as pd; pd.read_excel('large_file.xlsx').to_csv('converted.csv', index=False)"
   ```

### PDF Parsing Issues

**Issue**: PDF parsing produces garbled text or misses tables

**Solution**:
1. Ensure the PDF contains actual text, not just images of text

2. For scanned PDFs, ensure Tesseract OCR is properly installed

3. Try adjusting the PDF parsing settings in config/settings.yaml:
   ```yaml
   pdf_parser:
     use_ocr: true
     dpi: 300
     table_detection_confidence: 0.8
   ```

4. For complex PDFs, consider pre-processing with specialized tools like Tabula

## Field Mapping Issues

### Incorrect Field Mappings

**Issue**: The system incorrectly maps fields or has low confidence scores

**Solution**:
1. Provide more sample data to improve mapping accuracy

2. Manually adjust mappings through the UI or API:
   ```bash
   # Using the API
   curl -X POST -H "Content-Type: application/json" -d '{"mappings":{"source_column":"target_field"}}' http://localhost:8080/api/map/job_id
   ```

3. Create a mapping template file for similar catalogs:
   ```json
   {
     "Item #": "SKU",
     "Product Name": "Short Description",
     "Description": "Long Description"
   }
   ```

### Missing Field Mappings

**Issue**: Some fields are not mapped or are mapped to "Do not include"

**Solution**:
1. Check if the source columns exist in the input file

2. Ensure the column names are correctly parsed (check for whitespace or special characters)

3. Manually map the fields through the UI or API

4. Update the LLM prompts in prompts/templates/mapping_prompt.txt to improve recognition

## Performance Issues

### Slow Processing for Large Files

**Issue**: Processing large files takes too long

**Solution**:
1. Enable chunked processing:
   ```bash
   python app.py --input large_file.csv --chunk-size 10000
   ```

2. Increase the number of worker processes:
   ```bash
   python app.py --input large_file.csv --workers 8
   ```

3. Enable memory optimization:
   ```bash
   python app.py --input large_file.csv --optimize-memory
   ```

4. Use a more powerful machine or increase Docker container resources

### Memory Issues

**Issue**: Application crashes with "MemoryError" or "Out of memory"

**Solution**:
1. Enable chunked processing with smaller chunk sizes:
   ```bash
   python app.py --input large_file.csv --chunk-size 5000
   ```

2. Optimize DataFrame memory usage:
   ```bash
   python app.py --input large_file.csv --optimize-memory
   ```

3. Increase system swap space or Docker container memory limits

4. For very large files, use a database backend:
   ```bash
   python app.py --input large_file.csv --use-db
   ```

## API Issues

### API Connection Issues

**Issue**: Frontend cannot connect to the API

**Solution**:
1. Check if the API server is running:
   ```bash
   curl http://localhost:8080/api/health
   ```

2. Check for CORS issues in the browser console

3. Ensure the API URL is correctly configured in the frontend:
   ```javascript
   // In web/frontend/.env
   REACT_APP_API_URL=http://localhost:8080
   ```

4. Check for network or firewall issues

### API Rate Limiting Issues

**Issue**: API requests are being rate limited

**Solution**:
1. Implement backoff and retry logic in API clients

2. Adjust rate limiting settings in config/settings.yaml:
   ```yaml
   rate_limiting:
     max_requests: 100
     time_period: 60  # seconds
   ```

3. For high-volume usage, deploy multiple API instances behind a load balancer

## Frontend Issues

### UI Rendering Issues

**Issue**: UI components don't render correctly or are missing

**Solution**:
1. Clear browser cache and reload

2. Check browser console for JavaScript errors

3. Ensure all dependencies are correctly installed:
   ```bash
   cd web/frontend
   npm install
   ```

4. Try a different browser

### File Upload Issues

**Issue**: File upload fails or hangs

**Solution**:
1. Check file size limits in the API and frontend

2. Ensure the correct content type is being sent

3. Check network tab in browser developer tools for errors

4. Increase timeout settings for large file uploads:
   ```javascript
   // In web/frontend/src/services/api.js
   const apiClient = axios.create({
     timeout: 300000,  // 5 minutes
     // other settings
   });
   ```

## Docker Issues

### Container Startup Issues

**Issue**: Docker containers fail to start

**Solution**:
1. Check Docker logs:
   ```bash
   docker logs av-catalog-converter-api
   ```

2. Ensure ports are not already in use:
   ```bash
   lsof -i :8080
   lsof -i :3000
   ```

3. Check Docker Compose configuration:
   ```bash
   docker compose config
   ```

4. Ensure Docker has enough resources allocated

### Volume Mounting Issues

**Issue**: Docker container cannot access mounted volumes

**Solution**:
1. Check volume permissions:
   ```bash
   ls -la ./data
   ```

2. Ensure the paths are correct in docker-compose.yml

3. Try using absolute paths for volume mounts

4. For Windows users, ensure file sharing is enabled for the relevant drives

## LLM Integration Issues

### LLM Loading Issues

**Issue**: Error loading LLM model

**Solution**:
1. Check if the model files exist in the expected location

2. Ensure you have enough RAM for the model (at least 4GB for Phi-2)

3. Try using a quantized version of the model:
   ```yaml
   # In config/settings.yaml
   llm:
     model_id: microsoft/phi-2
     quantization: 4bit
   ```

4. For GPU acceleration, ensure CUDA is properly installed and configured

### LLM Generation Issues

**Issue**: LLM generates poor quality or incorrect mappings

**Solution**:
1. Adjust the temperature setting:
   ```yaml
   # In config/settings.yaml
   llm:
     temperature: 0.2  # Lower for more deterministic outputs
   ```

2. Improve the prompts in prompts/templates/

3. Provide more context or examples in the prompt

4. Consider using a different LLM model

## Common Error Messages

### "No module named 'X'"

**Issue**: Python module not found

**Solution**:
1. Install the missing module:
   ```bash
   pip install X
   ```

2. Check if the module is in requirements.txt

3. Ensure you're using the correct virtual environment

### "Permission denied"

**Issue**: Insufficient permissions to access files or directories

**Solution**:
1. Check file permissions:
   ```bash
   ls -la path/to/file
   ```

2. Change permissions if needed:
   ```bash
   chmod 644 path/to/file
   ```

3. For Docker, check user and group mappings

### "Address already in use"

**Issue**: Port is already being used by another process

**Solution**:
1. Find the process using the port:
   ```bash
   lsof -i :8080
   ```

2. Kill the process or use a different port:
   ```bash
   kill -9 <PID>
   # or
   python app.py --api --port 8081
   ```

For additional help, please check the [GitHub Issues](https://github.com/yourusername/av-catalog-converter/issues) or contact support.
