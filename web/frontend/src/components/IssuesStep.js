import React from 'react';
import {
  Box,
  Typography,
  Card,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Button,
  Divider,
  Alert,
  Stack
} from '@mui/material';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import InfoIcon from '@mui/icons-material/Info';

const IssuesStep = ({ issues, onNext, onPrevious }) => {
  // Count issues by type
  const errorCount = issues.filter(issue => issue.type === 'error').length;
  const warningCount = issues.filter(issue => issue.type === 'warning').length;
  const infoCount = issues.filter(issue => issue.type === 'info').length;

  // Get icon based on issue type
  const getIssueIcon = (type) => {
    switch (type) {
      case 'error':
        return <ErrorIcon color="error" />;
      case 'warning':
        return <WarningIcon color="warning" />;
      case 'info':
        return <InfoIcon color="info" />;
      default:
        return <InfoIcon />;
    }
  };

  // Get severity for Alert component
  const getAlertSeverity = (type) => {
    switch (type) {
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      default:
        return 'info';
    }
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        Issues and Warnings
      </Typography>
      <Typography variant="body1" paragraph>
        We've identified the following issues with your catalog data. Please review before proceeding.
      </Typography>

      <Stack direction="row" spacing={2} sx={{ mb: 3 }}>
        {errorCount > 0 && (
          <Alert severity="error" icon={<ErrorIcon />}>
            {errorCount} error{errorCount !== 1 ? 's' : ''}
          </Alert>
        )}
        {warningCount > 0 && (
          <Alert severity="warning" icon={<WarningIcon />}>
            {warningCount} warning{warningCount !== 1 ? 's' : ''}
          </Alert>
        )}
        {infoCount > 0 && (
          <Alert severity="info" icon={<InfoIcon />}>
            {infoCount} info message{infoCount !== 1 ? 's' : ''}
          </Alert>
        )}
        {issues.length === 0 && (
          <Alert severity="success">
            No issues found! Your catalog looks great.
          </Alert>
        )}
      </Stack>

      {issues.length > 0 && (
        <Card sx={{ mb: 4 }}>
          <List>
            {issues.map((issue, index) => (
              <React.Fragment key={index}>
                {index > 0 && <Divider component="li" />}
                <ListItem>
                  <ListItemIcon>
                    {getIssueIcon(issue.type)}
                  </ListItemIcon>
                  <ListItemText primary={issue.message} />
                </ListItem>
              </React.Fragment>
            ))}
          </List>
        </Card>
      )}

      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Button
          variant="outlined"
          onClick={onPrevious}
          size="large"
        >
          Back
        </Button>
        <Button
          variant="contained"
          color="primary"
          onClick={onNext}
          size="large"
          disabled={errorCount > 0}
        >
          Continue to Download
        </Button>
      </Box>
    </Box>
  );
};

export default IssuesStep;
