import React from 'react';
import { Box, Typography, Paper, Button, Grid, Divider } from '@mui/material';
import { FileInfo } from '../types';

interface FileInfoStepProps {
  fileInfo: FileInfo;
  onNext: () => void;
}

const FileInfoStep: React.FC<FileInfoStepProps> = ({ fileInfo, onNext }) => {
  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        File Information
      </Typography>
      <Typography variant="body1" paragraph>
        We've analyzed your file. Please review the information below and continue to the next step.
      </Typography>
      
      <Paper sx={{ p: 3, mb: 4 }}>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6}>
            <Typography variant="subtitle1" color="textSecondary">
              File Name
            </Typography>
            <Typography variant="body1" gutterBottom>
              {fileInfo.name}
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="subtitle1" color="textSecondary">
              File Type
            </Typography>
            <Typography variant="body1" gutterBottom>
              {fileInfo.type}
            </Typography>
          </Grid>
          <Grid item xs={12}>
            <Divider sx={{ my: 1 }} />
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="subtitle1" color="textSecondary">
              File Size
            </Typography>
            <Typography variant="body1" gutterBottom>
              {fileInfo.size.toFixed(2)} MB
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6}>
            <Typography variant="subtitle1" color="textSecondary">
              Products Found
            </Typography>
            <Typography variant="body1" gutterBottom>
              {fileInfo.productCount}
            </Typography>
          </Grid>
        </Grid>
      </Paper>
      
      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
        <Button 
          variant="contained" 
          color="primary" 
          onClick={onNext}
          size="large"
        >
          Continue to Field Mapping
        </Button>
      </Box>
    </Box>
  );
};

export default FileInfoStep;
