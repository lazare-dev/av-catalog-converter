import React from 'react';
import { 
  Card, 
  Typography, 
  Button, 
  Box, 
  List,
  ListItem,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import WarningIcon from '@mui/icons-material/Warning';
import ErrorIcon from '@mui/icons-material/Error';
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
  // Get icon based on issue type
  const getIssueIcon = (type: IssueType) => {
    switch (type) {
      case IssueType.WARNING:
        return <WarningIcon sx={{ color: 'warning.main' }} />;
      case IssueType.ERROR:
        return <ErrorIcon sx={{ color: 'error.main' }} />;
      case IssueType.INFO:
        return <InfoIcon sx={{ color: 'info.main' }} />;
      default:
        return <InfoIcon />;
    }
  };

  return (
    <Card>
      <Typography variant="h2" gutterBottom>
        Issues & Warnings
      </Typography>
      
      {issues.length === 0 ? (
        <Typography variant="body1" sx={{ mb: 3 }}>
          No issues found. Your catalog is ready for processing.
        </Typography>
      ) : (
        <List sx={{ mb: 3 }}>
          {issues.map((issue, index) => (
            <ListItem key={index}>
              <ListItemIcon>
                {getIssueIcon(issue.type)}
              </ListItemIcon>
              <ListItemText primary={issue.message} />
            </ListItem>
          ))}
        </List>
      )}
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Button 
          variant="outlined" 
          onClick={onPrevious}
        >
          Back
        </Button>
        <Button 
          variant="contained" 
          onClick={onNext}
          color={issues.some(i => i.type === IssueType.ERROR) ? 'warning' : 'primary'}
        >
          {issues.some(i => i.type === IssueType.ERROR) 
            ? 'Continue with Errors' 
            : 'Continue to Download'}
        </Button>
      </Box>
    </Card>
  );
};

export default IssuesStep;
