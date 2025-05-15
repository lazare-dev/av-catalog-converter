import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Button,
  Typography,
  Paper,
  Stepper,
  Step,
  StepLabel,
  CircularProgress,
  Alert,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Snackbar,
  Grid,
  Divider,
  Chip
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import DownloadIcon from '@mui/icons-material/Download';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import InfoIcon from '@mui/icons-material/Info';
import { logger } from '../services/logging';
import { apiService } from '../services/api';

// Steps in the conversion process
const steps = ['Upload File', 'Validate', 'Export'];

// File validation status
enum ValidationStatus {
  NONE = 'none',
  VALIDATING = 'validating',
  VALID = 'valid',
  INVALID = 'invalid'
}

// Interface for file information
interface FileInfo {
  name: string;
  type: string;
  size: number;
  productCount: number;
}

// Interface for validation issues
interface ValidationIssue {
  type: 'error' | 'warning' | 'info';
  message: string;
}

const SimpleCatalogConverter: React.FC = () => {
  // Component state
  const [activeStep, setActiveStep] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [fileInfo, setFileInfo] = useState<FileInfo | null>(null);
  const [validationStatus, setValidationStatus] = useState<ValidationStatus>(ValidationStatus.NONE);
  const [validationIssues, setValidationIssues] = useState<ValidationIssue[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [outputFormat, setOutputFormat] = useState<string>('csv');
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');
  const [snackbarSeverity, setSnackbarSeverity] = useState<'success' | 'error' | 'info' | 'warning'>('info');

  // File input ref
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Log component mount
  useEffect(() => {
    logger.info('SimpleCatalogConverter', 'Component mounted');
    return () => {
      logger.info('SimpleCatalogConverter', 'Component unmounted');
    };
  }, []);

  // Handle file selection
  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) {
      logger.warning('SimpleCatalogConverter', 'No file selected');
      return;
    }

    const selectedFile = files[0];
    logger.info('SimpleCatalogConverter', 'File selected', {
      fileName: selectedFile.name,
      fileSize: selectedFile.size,
      fileType: selectedFile.type
    });

    setFile(selectedFile);
    setValidationStatus(ValidationStatus.NONE);
    setValidationIssues([]);
    setDownloadUrl(null);
  };

  // Handle file upload button click
  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Validate the selected file
  const validateFile = async () => {
    if (!file) {
      logger.warning('SimpleCatalogConverter', 'No file to validate');
      showSnackbar('Please select a file first', 'error');
      return;
    }

    logger.info('SimpleCatalogConverter', 'Starting file validation', { fileName: file.name });
    setValidationStatus(ValidationStatus.VALIDATING);
    setIsProcessing(true);

    try {
      // Call API to analyze file
      const { fileInfo: info, structure } = await apiService.analyzeFile(file);

      // Ensure we have valid info and structure objects
      if (!info) {
        throw new Error('File analysis returned no file information');
      }

      setFileInfo(info);
      logger.info('SimpleCatalogConverter', 'File analysis complete', { fileInfo: info });

      // Generate validation issues based on structure
      const issues: ValidationIssue[] = [];

      // Check for product count
      if (info.productCount === 0) {
        issues.push({
          type: 'error',
          message: 'No products detected in the file'
        });
      } else if (info.productCount < 5) {
        issues.push({
          type: 'warning',
          message: `Only ${info.productCount} products detected, which is unusually low`
        });
      } else {
        issues.push({
          type: 'info',
          message: `${info.productCount} products detected in the file`
        });
      }

      // Check file size
      if (info.size > 50) { // Size in MB
        issues.push({
          type: 'warning',
          message: `File size is large (${info.size.toFixed(2)} MB), processing may take longer`
        });
      }

      // Safely check for required columns
      const requiredColumns = ['sku', 'name', 'price'];

      // Make sure structure exists and has columns
      const structureObj = structure || {};

      // Make sure structure.columns exists and is an array
      const columns = Array.isArray(structureObj.columns) ? structureObj.columns :
                     (Array.isArray(structureObj.column_names) ? structureObj.column_names : []);

      const missingColumns = requiredColumns.filter(col =>
        !columns.some((c: string) =>
          c && typeof c === 'string' && c.toLowerCase().includes(col.toLowerCase())
        )
      );

      if (missingColumns.length > 0) {
        issues.push({
          type: 'warning',
          message: `Possible missing columns: ${missingColumns.join(', ')}`
        });
      }

      setValidationIssues(issues);

      // Set validation status based on issues
      const hasErrors = issues.some(issue => issue.type === 'error');
      setValidationStatus(hasErrors ? ValidationStatus.INVALID : ValidationStatus.VALID);

      // Move to next step if valid
      if (!hasErrors) {
        setActiveStep(1);
        showSnackbar('File validated successfully', 'success');
      } else {
        showSnackbar('File validation failed', 'error');
      }

    } catch (error: any) {
      logger.error('SimpleCatalogConverter', 'File validation failed', {
        error: error?.message || 'Unknown error',
        stack: error?.stack
      });

      setValidationStatus(ValidationStatus.INVALID);

      // Create a more descriptive error message
      const errorMessage = error?.message || 'Unknown error occurred';
      setValidationIssues([{
        type: 'error',
        message: `Failed to validate file: ${errorMessage}. Please try a different file.`
      }]);

      showSnackbar(`File validation failed: ${errorMessage}`, 'error');
    } finally {
      setIsProcessing(false);
    }
  };

  // Process and export the file
  const processFile = async () => {
    if (!file) {
      logger.warning('SimpleCatalogConverter', 'No file to process');
      showSnackbar('Please select a file first', 'error');
      return;
    }

    logger.info('SimpleCatalogConverter', 'Starting file processing', {
      fileName: file.name,
      outputFormat
    });

    setIsProcessing(true);

    try {
      // Call API to process file
      const url = await apiService.processFile(file, outputFormat);

      if (!url) {
        throw new Error('No download URL returned from the server');
      }

      setDownloadUrl(url);
      setActiveStep(2);
      logger.info('SimpleCatalogConverter', 'File processing complete', { downloadUrl: url });
      showSnackbar('File processed successfully', 'success');
    } catch (error: any) {
      logger.error('SimpleCatalogConverter', 'File processing failed', {
        error: error?.message || 'Unknown error',
        stack: error?.stack
      });

      // Create a more descriptive error message
      const errorMessage = error?.message || 'Unknown error occurred';
      showSnackbar(`Failed to process file: ${errorMessage}`, 'error');

      // Add validation issue
      setValidationIssues(prev => [
        ...prev,
        {
          type: 'error',
          message: `Processing failed: ${errorMessage}`
        }
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle download
  const handleDownload = () => {
    if (!downloadUrl) {
      logger.warning('SimpleCatalogConverter', 'No download URL available');
      return;
    }

    logger.info('SimpleCatalogConverter', 'Downloading file');

    // Create a temporary link and click it
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `converted_catalog.${outputFormat}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Reset the form to start over
  const handleReset = () => {
    logger.info('SimpleCatalogConverter', 'Resetting form');
    setFile(null);
    setFileInfo(null);
    setValidationStatus(ValidationStatus.NONE);
    setValidationIssues([]);
    setDownloadUrl(null);
    setActiveStep(0);

    // Reset file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Show snackbar message
  const showSnackbar = (message: string, severity: 'success' | 'error' | 'info' | 'warning') => {
    setSnackbarMessage(message);
    setSnackbarSeverity(severity);
    setSnackbarOpen(true);
  };

  // Handle snackbar close
  const handleSnackbarClose = () => {
    setSnackbarOpen(false);
  };

  return (
    <Box sx={{ width: '100%', p: 3 }}>
      <Paper elevation={3} sx={{ p: 4, mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          AV Catalog Converter - Updated Version
        </Typography>
        <Typography variant="subtitle1" gutterBottom align="center" color="text.secondary">
          Convert your audio-visual equipment catalog to a standardized format
        </Typography>

        <Stepper activeStep={activeStep} sx={{ mb: 4, mt: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        <Box sx={{ mt: 4 }}>
          {activeStep === 0 && (
            <Box sx={{ textAlign: 'center' }}>
              <input
                type="file"
                accept=".csv,.xlsx,.xls,.json,.xml"
                style={{ display: 'none' }}
                onChange={handleFileSelect}
                ref={fileInputRef}
              />

              <Button
                variant="contained"
                startIcon={<CloudUploadIcon />}
                onClick={handleUploadClick}
                size="large"
                sx={{ mb: 3 }}
              >
                Select Catalog File
              </Button>

              {file && (
                <Box sx={{ mt: 2, mb: 3 }}>
                  <Typography variant="body1">
                    Selected file: <strong>{file.name}</strong> ({(file.size / (1024 * 1024)).toFixed(2)} MB)
                  </Typography>

                  <Box sx={{ mt: 3 }}>
                    <FormControl sx={{ minWidth: 200, mr: 2 }}>
                      <InputLabel id="output-format-label">Output Format</InputLabel>
                      <Select
                        labelId="output-format-label"
                        value={outputFormat}
                        label="Output Format"
                        onChange={(e) => setOutputFormat(e.target.value)}
                      >
                        <MenuItem value="csv">CSV</MenuItem>
                        <MenuItem value="excel">Excel</MenuItem>
                        <MenuItem value="json">JSON</MenuItem>
                      </Select>
                    </FormControl>

                    <Button
                      variant="contained"
                      onClick={validateFile}
                      disabled={isProcessing}
                      sx={{ mt: 1 }}
                    >
                      {isProcessing ? <CircularProgress size={24} /> : 'Validate File'}
                    </Button>
                  </Box>
                </Box>
              )}
            </Box>
          )}

          {activeStep === 1 && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Validation Results
              </Typography>

              <Grid container spacing={2} sx={{ mb: 3 }}>
                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      File Information
                    </Typography>
                    {fileInfo && (
                      <Box>
                        <Typography variant="body2">Name: {fileInfo.name}</Typography>
                        <Typography variant="body2">Type: {fileInfo.type}</Typography>
                        <Typography variant="body2">Size: {fileInfo.size.toFixed(2)} MB</Typography>
                        <Typography variant="body2">Products: {fileInfo.productCount}</Typography>
                      </Box>
                    )}
                  </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Paper variant="outlined" sx={{ p: 2 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      Validation Issues
                    </Typography>
                    {validationIssues.length === 0 ? (
                      <Typography variant="body2">No issues found</Typography>
                    ) : (
                      <Box>
                        {validationIssues.map((issue, index) => (
                          <Alert
                            key={index}
                            severity={issue.type}
                            sx={{ mb: 1 }}
                            icon={
                              issue.type === 'error' ? <ErrorIcon /> :
                              issue.type === 'warning' ? <ErrorIcon color="warning" /> :
                              <InfoIcon />
                            }
                          >
                            {issue.message}
                          </Alert>
                        ))}
                      </Box>
                    )}
                  </Paper>
                </Grid>
              </Grid>

              <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
                <Button onClick={() => setActiveStep(0)}>
                  Back
                </Button>
                <Button
                  variant="contained"
                  onClick={processFile}
                  disabled={isProcessing || validationStatus === ValidationStatus.INVALID}
                >
                  {isProcessing ? <CircularProgress size={24} /> : 'Process & Export'}
                </Button>
              </Box>
            </Box>
          )}

          {activeStep === 2 && (
            <Box sx={{ textAlign: 'center' }}>
              <CheckCircleIcon color="success" sx={{ fontSize: 60, mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Processing Complete!
              </Typography>

              <Typography variant="body1" paragraph>
                Your catalog has been successfully converted to {outputFormat.toUpperCase()} format.
              </Typography>

              <Button
                variant="contained"
                startIcon={<DownloadIcon />}
                onClick={handleDownload}
                sx={{ mt: 2 }}
              >
                Download Converted File
              </Button>

              <Box sx={{ mt: 4 }}>
                <Button onClick={handleReset}>
                  Convert Another File
                </Button>
              </Box>
            </Box>
          )}
        </Box>
      </Paper>

      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleSnackbarClose} severity={snackbarSeverity}>
          {snackbarMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default SimpleCatalogConverter;
