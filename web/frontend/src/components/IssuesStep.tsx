import React from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Button, 
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider
} from '@mui/material';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import InfoIcon from '@mui/icons-material/Info';
import { Issue, IssueType } from '../types';

interface IssuesStepProps {
  issues: Issue[];
  onNext: () => void;
  onPrevious: () => void;
}

const IssuesStep: React.FC<IssuesStepProps> = ({ 
  issues, 
  onNext, 
  onPrevious 
}) => {
  const getIssueIcon = (type: IssueType) => {
    switch (type) {
      case IssueType.ERROR:
        return <ErrorIcon color="error" />;
      case IssueType.WARNING:
        return <WarningIcon color="warning" />;
      case IssueType.INFO:
        return <InfoIcon color="info" />;
      default:
        return <InfoIcon color="info" />;
    }
  };

  const errorCount = issues.filter(issue => issue.type === IssueType.ERROR).length;
  const warningCount = issues.filter(issue => issue.type === IssueType.WARNING).length;
  const infoCount = issues.filter(issue => issue.type === IssueType.INFO).length;

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        Issues and Warnings
      </Typography>
      <Typography variant="body1" paragraph>
        We've identified the following issues with your catalog. Errors must be fixed before proceeding, 
        while warnings and information items are optional to address.
      </Typography>
      
      <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
        <Paper sx={{ p: 2, flex: 1, backgroundColor: errorCount > 0 ? 'rgba(231, 76, 60, 0.1)' : 'inherit' }}>
          <Typography variant="h6" color="error" gutterBottom>
            Errors
          </Typography>
          <Typography variant="h3">
            {errorCount}
          </Typography>
        </Paper>
        <Paper sx={{ p: 2, flex: 1, backgroundColor: warningCount > 0 ? 'rgba(243, 156, 18, 0.1)' : 'inherit' }}>
          <Typography variant="h6" color="warning" gutterBottom>
            Warnings
          </Typography>
          <Typography variant="h3">
            {warningCount}
          </Typography>
        </Paper>
        <Paper sx={{ p: 2, flex: 1, backgroundColor: infoCount > 0 ? 'rgba(52, 152, 219, 0.1)' : 'inherit' }}>
          <Typography variant="h6" color="info" gutterBottom>
            Information
          </Typography>
          <Typography variant="h3">
            {infoCount}
          </Typography>
        </Paper>
      </Box>
      
      <Paper sx={{ mb: 4 }}>
        <List>
          {issues.map((issue, index) => (
            <React.Fragment key={index}>
              {index > 0 && <Divider />}
              <ListItem>
                <ListItemIcon>
                  {getIssueIcon(issue.type)}
                </ListItemIcon>
                <ListItemText 
                  primary={issue.message}
                  secondary={issue.type.charAt(0).toUpperCase() + issue.type.slice(1)}
                />
              </ListItem>
            </React.Fragment>
          ))}
          {issues.length === 0 && (
            <ListItem>
              <ListItemText 
                primary="No issues found"
                secondary="Your catalog looks good!"
              />
            </ListItem>
          )}
        </List>
      </Paper>
      
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
