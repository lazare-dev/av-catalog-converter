import React from 'react';
import { Box, Typography, Card, CardContent, Button, Grid } from '@mui/material';
import DescriptionIcon from '@mui/icons-material/Description';
import StorageIcon from '@mui/icons-material/Storage';
import CategoryIcon from '@mui/icons-material/Category';
import FileSizeIcon from '@mui/icons-material/InsertDriveFile';

const FileInfoStep = ({ fileInfo, onNext }) => {
  if (!fileInfo) {
    return <Typography>No file information available</Typography>;
  }

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        File Information
      </Typography>
      <Typography variant="body1" paragraph>
        We've analyzed your file. Please review the information below and continue to the next step.
      </Typography>

      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <DescriptionIcon sx={{ mr: 2, color: 'primary.main' }} />
                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    File Name
                  </Typography>
                  <Typography variant="body1">
                    {fileInfo.name}
                  </Typography>
                </Box>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <CategoryIcon sx={{ mr: 2, color: 'primary.main' }} />
                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    File Type
                  </Typography>
                  <Typography variant="body1">
                    {fileInfo.type}
                  </Typography>
                </Box>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <FileSizeIcon sx={{ mr: 2, color: 'primary.main' }} />
                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    File Size
                  </Typography>
                  <Typography variant="body1">
                    {fileInfo.size.toFixed(2)} MB
                  </Typography>
                </Box>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <StorageIcon sx={{ mr: 2, color: 'primary.main' }} />
                <Box>
                  <Typography variant="subtitle2" color="textSecondary">
                    Product Count
                  </Typography>
                  <Typography variant="body1">
                    {fileInfo.productCount} products
                  </Typography>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

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
