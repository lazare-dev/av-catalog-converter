import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Card, Typography, Button, Box, CircularProgress } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

interface UploadStepProps {
  onFileUpload: (file: File) => void;
}

const UploadStep: React.FC<UploadStepProps> = ({ onFileUpload }) => {
  const [isUploading, setIsUploading] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setIsUploading(true);
      onFileUpload(acceptedFiles[0]);
    }
  }, [onFileUpload]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/json': ['.json'],
      'application/xml': ['.xml'],
      'application/pdf': ['.pdf']
    },
    multiple: false
  });

  return (
    <Card>
      {isUploading ? (
        <Box sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          p: 5,
          minHeight: 300
        }}>
          <CircularProgress size={60} sx={{ mb: 3 }} />
          <Typography variant="h6" gutterBottom>
            Uploading and Analyzing File...
          </Typography>
          <Typography variant="body2" color="textSecondary">
            This may take a moment depending on the file size.
          </Typography>
        </Box>
      ) : (
        <Box
          {...getRootProps()}
          sx={{
            border: '2px dashed',
            borderColor: isDragActive ? 'primary.main' : 'grey.300',
            borderRadius: 2,
            p: 5,
            textAlign: 'center',
            cursor: 'pointer',
            backgroundColor: isDragActive ? 'rgba(52, 152, 219, 0.05)' : 'transparent',
            transition: 'all 0.3s',
            '&:hover': {
              borderColor: 'primary.main',
              backgroundColor: 'rgba(52, 152, 219, 0.05)'
            }
          }}
        >
          <input {...getInputProps()} />
          <CloudUploadIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Drag and drop your catalog file here
          </Typography>
          <Typography variant="body1" color="textSecondary" gutterBottom>
            or
          </Typography>
          <Button variant="contained" component="span">
            Browse Files
          </Button>
          <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
            Supported formats: CSV, Excel, PDF, JSON, XML
          </Typography>
        </Box>
      )}
    </Card>
  );
};

export default UploadStep;
