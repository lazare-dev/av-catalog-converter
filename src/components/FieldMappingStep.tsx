import React, { useState } from 'react';
import { 
  Card, 
  Typography, 
  Button, 
  Box, 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip
} from '@mui/material';
import { FieldMapping } from '../types';

interface FieldMappingStepProps {
  fieldMappings: FieldMapping[];
  onUpdate: (mappings: FieldMapping[]) => void;
  onNext: () => void;
  onPrevious: () => void;
}

// Standard field options
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

const FieldMappingStep: React.FC<FieldMappingStepProps> = ({ 
  fieldMappings, 
  onUpdate, 
  onNext, 
  onPrevious 
}) => {
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');

  // Handle field mapping change
  const handleMappingChange = (index: number, newValue: string) => {
    const updatedMappings = [...fieldMappings];
    updatedMappings[index] = {
      ...updatedMappings[index],
      mapsTo: newValue
    };
    onUpdate(updatedMappings);
  };

  // Handle template selection
  const handleTemplateChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    setSelectedTemplate(event.target.value as string);
    // In a real app, this would load a predefined mapping template
  };

  // Render confidence chip
  const renderConfidenceChip = (confidence: number) => {
    if (confidence === 0) return null;
    
    let color = 'success';
    if (confidence < 80) color = 'warning';
    if (confidence < 60) color = 'error';
    
    return (
      <Chip 
        label={`${confidence}%`} 
        color={color as 'success' | 'warning' | 'error'} 
        size="small" 
        sx={{ borderRadius: '12px', fontSize: '12px' }}
      />
    );
  };

  return (
    <Card>
      <Typography variant="h2" gutterBottom>
        Field Mapping
      </Typography>
      
      <TableContainer sx={{ mb: 3 }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Source Column</TableCell>
              <TableCell>Maps To</TableCell>
              <TableCell>Confidence</TableCell>
              <TableCell>Sample</TableCell>
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
                  {renderConfidenceChip(mapping.confidence)}
                </TableCell>
                <TableCell>{mapping.sample}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Button variant="outlined" onClick={() => {}}>
          Edit Mappings
        </Button>
        <FormControl sx={{ minWidth: 200 }}>
          <InputLabel id="template-label">Apply Saved Template</InputLabel>
          <Select
            labelId="template-label"
            value={selectedTemplate}
            onChange={(e) => handleTemplateChange(e as any)}
            label="Apply Saved Template"
          >
            <MenuItem value="">
              <em>None</em>
            </MenuItem>
            <MenuItem value="audio">Audio Supplier Template</MenuItem>
            <MenuItem value="video">Video Equipment Template</MenuItem>
            <MenuItem value="general">General AV Template</MenuItem>
          </Select>
        </FormControl>
      </Box>
      
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
        >
          Continue to Standardization
        </Button>
      </Box>
    </Card>
  );
};

export default FieldMappingStep;
