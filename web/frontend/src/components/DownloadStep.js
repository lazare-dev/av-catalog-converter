import React from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  Button, 
  CircularProgress,
  Link,
  Stack,
  Divider
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CloudDownloadIcon from '@mui/icons-material/CloudDownload';

const DownloadStep = ({ isProcessing, downloadUrl, onProcess, onPrevious }) => {
  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        Process and Download
      </Typography>
      <Typography variant="body1" paragraph>
        Your catalog is ready to be processed. Click the button below to generate your standardized catalog.
      </Typography>

      <Card sx={{ mb: 4 }}>
        <CardContent sx={{ textAlign: 'center', py: 4 }}>
          {isProcessing ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <CircularProgress size={60} sx={{ mb: 2 }} />
              <Typography variant="h6">
                Processing your catalog...
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                This may take a few moments depending on the size of your catalog.
              </Typography>
            </Box>
          ) : downloadUrl ? (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <CheckCircleIcon color="success" sx={{ fontSize: 60, mb: 2 }} />
              <Typography variant="h6" gutterBottom>
                Your catalog has been processed successfully!
              </Typography>
              <Button
                variant="contained"
                color="primary"
                startIcon={<CloudDownloadIcon />}
                component={Link}
                href={downloadUrl}
                download
                sx={{ mt: 2 }}
                size="large"
              >
                Download Standardized Catalog
              </Button>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
                Your file will be available for download for the next 24 hours.
              </Typography>
            </Box>
          ) : (
            <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <Button
                variant="contained"
                color="primary"
                onClick={onProcess}
                size="large"
                sx={{ py: 1.5, px: 4 }}
              >
                Process Catalog
              </Button>
              <Typography variant="body2" color="textSecondary" sx={{ mt: 2 }}>
                This will apply all your selected options and generate the final catalog.
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {!isProcessing && (
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
      )}
    </Box>
  );
};

export default DownloadStep;
