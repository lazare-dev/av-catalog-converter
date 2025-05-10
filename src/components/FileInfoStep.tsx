import React from 'react';
import { Card, Typography, Button, Grid, Box, CircularProgress } from '@mui/material';
import { FileInfo } from '../types';

interface FileInfoStepProps {
  fileInfo: FileInfo | null;
  onNext: () => void;
  isLoading?: boolean;
}

const FileInfoStep: React.FC<FileInfoStepProps> = ({
  fileInfo,
  onNext,
  isLoading = false
}) => {
  return (
    <Card>
      <Typography variant="h2" gutterBottom>
        File Information
      </Typography>

      {isLoading || !fileInfo ? (
        <Box sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          p: 5,
          minHeight: 200
        }}>
          <CircularProgress size={40} sx={{ mb: 2 }} />
          <Typography variant="body1">
            Analyzing file structure...
          </Typography>
        </Box>
      ) : (
        <>
          <Grid container spacing={3} sx={{ mb: 4 }}>
            <Grid item xs={12} md={6}>
              <Typography variant="body1">
                <strong>File:</strong> {fileInfo.name}
              </Typography>
              <Typography variant="body1">
                <strong>Type:</strong> {fileInfo.type}
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="body1">
                <strong>Size:</strong> {fileInfo.size.toFixed(2)} MB
              </Typography>
              <Typography variant="body1">
                <strong>Products:</strong> {fileInfo.productCount}
              </Typography>
            </Grid>
          </Grid>

          <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              variant="contained"
              onClick={onNext}
            >
              Continue to Field Mapping
            </Button>
          </Box>
        </>
      )}
    </Card>
  );
};

export default FileInfoStep;
