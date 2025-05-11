import React from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Button, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow,
  Select,
  MenuItem,
  FormControl,
  SelectChangeEvent,
  Chip
} from '@mui/material';
import { FieldMapping } from '../types';

interface FieldMappingStepProps {
  fieldMappings: FieldMapping[];
  onUpdate: (updatedMappings: FieldMapping[]) => void;
  onNext: () => void;
  onPrevious: () => void;
}

const FieldMappingStep: React.FC<FieldMappingStepProps> = ({ 
  fieldMappings, 
  onUpdate, 
  onNext, 
  onPrevious 
}) => {
  const standardFields = [
    'Product Name',
    'SKU',
    'Price',
    'Product Category',
    'Manufacturer',
    'Description',
    'Specifications',
    'Do not include'
  ];

  const handleMappingChange = (index: number, value: string) => {
    const updatedMappings = [...fieldMappings];
    updatedMappings[index] = {
      ...updatedMappings[index],
      mapsTo: value
    };
    onUpdate(updatedMappings);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'success';
    if (confidence >= 70) return 'info';
    if (confidence >= 50) return 'warning';
    return 'error';
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        Field Mapping
      </Typography>
      <Typography variant="body1" paragraph>
        We've automatically mapped your catalog fields to our standard fields. Please review and adjust if needed.
      </Typography>
      
      <TableContainer component={Paper} sx={{ mb: 4 }}>
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
            {fieldMappings.map((mapping, index) => (
              <TableRow key={index}>
                <TableCell>{mapping.sourceColumn}</TableCell>
                <TableCell>
                  <FormControl fullWidth size="small">
                    <Select
                      value={mapping.mapsTo}
                      onChange={(e: SelectChangeEvent) => handleMappingChange(index, e.target.value)}
                    >
                      {standardFields.map((field) => (
                        <MenuItem key={field} value={field}>{field}</MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </TableCell>
                <TableCell>
                  <Chip 
                    label={`${mapping.confidence}%`} 
                    color={getConfidenceColor(mapping.confidence) as any}
                    size="small"
                    sx={{ minWidth: 70 }}
                  />
                </TableCell>
                <TableCell>{mapping.sample}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      
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
