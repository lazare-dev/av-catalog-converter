import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Box, Typography, Paper, Button } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const UploadStep = ({ onFileUpload }) => {
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      onFileUpload(acceptedFiles[0]);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/pdf': ['.pdf'],
      'application/json': ['.json'],
      'application/xml': ['.xml'],
    },
    multiple: false
  });

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        Upload Catalog
      </Typography>
      <Typography variant="body1" paragraph>
        Upload your catalog file to begin the conversion process. We support CSV, Excel, PDF, JSON, and XML formats.
      </Typography>
      
      <Paper
        {...getRootProps()}
        sx={{
          p: 6,
          textAlign: 'center',
          cursor: 'pointer',
          backgroundColor: isDragActive ? 'rgba(52, 152, 219, 0.1)' : 'white',
          border: isDragActive ? '2px dashed #3498db' : '2px dashed #e0e0e0',
          borderRadius: 2,
          transition: 'all 0.3s ease',
          '&:hover': {
            backgroundColor: 'rgba(52, 152, 219, 0.05)',
            border: '2px dashed #3498db'
          }
        }}
      >
        <input {...getInputProps()} />
        <CloudUploadIcon sx={{ fontSize: 60, color: '#3498db', mb: 2 }} />
        <Typography variant="h5" gutterBottom>
          {isDragActive ? 'Drop the file here' : 'Drag & drop a file here'}
        </Typography>
        <Typography variant="body1" color="textSecondary" paragraph>
          or
        </Typography>
        <Button variant="contained" component="span">
          Browse Files
        </Button>
        <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
          Supported formats: CSV, Excel, PDF, JSON, XML
        </Typography>
      </Paper>
    </Box>
  );
};

export default UploadStep;
