import React from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Button, 
  CircularProgress,
  Link
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import DownloadIcon from '@mui/icons-material/Download';

interface DownloadStepProps {
  isProcessing: boolean;
  downloadUrl: string | null;
  onProcess: () => void;
  onPrevious: () => void;
}

const DownloadStep: React.FC<DownloadStepProps> = ({ 
  isProcessing, 
  downloadUrl, 
  onProcess, 
  onPrevious 
}) => {
  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        Process and Download
      </Typography>
      <Typography variant="body1" paragraph>
        Your catalog is ready to be processed. Click the button below to start processing and download the standardized catalog.
      </Typography>
      
      <Paper sx={{ p: 4, textAlign: 'center', mb: 4 }}>
        {isProcessing ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <CircularProgress size={60} sx={{ mb: 2 }} />
            <Typography variant="h6">
              Processing your catalog...
            </Typography>
            <Typography variant="body2" color="textSecondary">
              This may take a few moments depending on the size of your catalog.
            </Typography>
          </Box>
        ) : downloadUrl ? (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <CheckCircleIcon color="success" sx={{ fontSize: 60, mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Processing Complete!
            </Typography>
            <Typography variant="body1" paragraph>
              Your standardized catalog is ready for download.
            </Typography>
            <Button 
              variant="contained" 
              color="primary" 
              startIcon={<DownloadIcon />}
              component={Link}
              href={downloadUrl}
              download
              size="large"
            >
              Download Standardized Catalog
            </Button>
          </Box>
        ) : (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Typography variant="h6" gutterBottom>
              Ready to Process
            </Typography>
            <Typography variant="body1" paragraph>
              Click the button below to process your catalog with the selected options.
            </Typography>
            <Button 
              variant="contained" 
              color="primary" 
              onClick={onProcess}
              size="large"
            >
              Process Catalog
            </Button>
          </Box>
        )}
      </Paper>
      
      <Box sx={{ display: 'flex', justifyContent: 'flex-start' }}>
        <Button 
          variant="outlined" 
          onClick={onPrevious}
          size="large"
          disabled={isProcessing}
        >
          Back
        </Button>
      </Box>
    </Box>
  );
};

export default DownloadStep;
