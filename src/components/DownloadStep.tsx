import React from 'react';
import {
  Card,
  Typography,
  Button,
  Box,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  Link
} from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import SaveIcon from '@mui/icons-material/Save';
import GetAppIcon from '@mui/icons-material/GetApp';

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
  // Processing steps
  const steps = [
    'Parsing file',
    'Mapping fields',
    'Normalizing values',
    'Extracting categories',
    'Generating output'
  ];

  return (
    <Card>
      <Typography variant="h2" gutterBottom>
        Process & Download
      </Typography>

      {isProcessing ? (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <CircularProgress size={60} sx={{ mb: 3 }} />
          <Typography variant="h6" gutterBottom>
            Processing Your Catalog
          </Typography>
          <Typography variant="body2" color="textSecondary" gutterBottom>
            This may take a few moments...
          </Typography>

          <Box sx={{ width: '100%', mt: 4 }}>
            <Stepper activeStep={2} alternativeLabel>
              {steps.map((label, index) => (
                <Step key={label} completed={index < 2}>
                  <StepLabel>{label}</StepLabel>
                </Step>
              ))}
            </Stepper>
          </Box>
        </Box>
      ) : downloadUrl ? (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <CheckCircleIcon sx={{ fontSize: 60, color: 'success.main', mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Processing Complete!
          </Typography>
          <Typography variant="body1" gutterBottom>
            Your catalog has been successfully processed and is ready for download.
          </Typography>

          <Box sx={{ mt: 4, mb: 2 }}>
            <Button
              variant="contained"
              color="secondary"
              startIcon={<GetAppIcon />}
              href={downloadUrl}
              download
              size="large"
            >
              Download CSV
            </Button>
          </Box>

          <Box sx={{ mt: 2 }}>
            <Link href="#" onClick={() => {}}>
              Save this configuration as a template
            </Link>
          </Box>
        </Box>
      ) : (
        <Box sx={{ py: 3 }}>
          <Typography variant="body1" paragraph>
            Your catalog is ready to be processed. Click the button below to start the conversion process.
          </Typography>
          <Typography variant="body2" color="textSecondary" paragraph>
            This will apply all the mappings and standardization options you've selected.
          </Typography>

          <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 4 }}>
            <Button
              variant="outlined"
              startIcon={<SaveIcon />}
              onClick={() => {}}
            >
              Save Template
            </Button>
            <Button
              variant="contained"
              color="primary"
              onClick={onProcess}
              size="large"
            >
              Process Catalog
            </Button>
          </Box>
        </Box>
      )}

      {!isProcessing && !downloadUrl && (
        <Box sx={{ display: 'flex', justifyContent: 'flex-start', mt: 2 }}>
          <Button
            variant="text"
            onClick={onPrevious}
          >
            Back to Issues
          </Button>
        </Box>
      )}
    </Card>
  );
};

export default DownloadStep;
