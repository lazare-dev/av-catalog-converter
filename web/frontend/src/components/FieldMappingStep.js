import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  Paper, 
  Select, 
  MenuItem, 
  FormControl, 
  Button, 
  Chip,
  Stack
} from '@mui/material';

const FieldMappingStep = ({ fieldMappings, onUpdate, onNext, onPrevious }) => {
  const [mappings, setMappings] = useState(fieldMappings);

  const handleMappingChange = (index, value) => {
    const updatedMappings = [...mappings];
    updatedMappings[index] = {
      ...updatedMappings[index],
      mapsTo: value
    };
    setMappings(updatedMappings);
    onUpdate(updatedMappings);
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 90) return 'success';
    if (confidence >= 70) return 'info';
    if (confidence >= 50) return 'warning';
    return 'error';
  };

  const standardFields = [
    'SKU',
    'Product Name',
    'Price',
    'Product Category',
    'Manufacturer',
    'Description',
    'Specifications',
    'Do not include'
  ];

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        Field Mapping
      </Typography>
      <Typography variant="body1" paragraph>
        We've automatically mapped your file's columns to our standard fields. Please review and adjust if needed.
      </Typography>

      <Card sx={{ mb: 4 }}>
        <TableContainer component={Paper}>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Source Column</TableCell>
                <TableCell>Maps To</TableCell>
                <TableCell>Confidence</TableCell>
                <TableCell>Sample Data</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {mappings.map((mapping, index) => (
                <TableRow key={index}>
                  <TableCell>{mapping.sourceColumn}</TableCell>
                  <TableCell>
                    <FormControl fullWidth size="small">
                      <Select
                        value={mapping.mapsTo}
                        onChange={(e) => handleMappingChange(index, e.target.value)}
                      >
                        {standardFields.map((field) => (
                          <MenuItem key={field} value={field}>
                            {field}
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>
                  </TableCell>
                  <TableCell>
                    <Chip 
                      label={`${mapping.confidence}%`} 
                      color={getConfidenceColor(mapping.confidence)}
                      size="small"
                    />
                  </TableCell>
                  <TableCell>{mapping.sample}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Card>

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
        >
          Continue to Standardization
        </Button>
      </Box>
    </Box>
  );
};

export default FieldMappingStep;
