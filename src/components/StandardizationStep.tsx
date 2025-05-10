import React from 'react';
import { 
  Card, 
  Typography, 
  Button, 
  Box, 
  Grid, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem,
  FormControlLabel,
  Checkbox
} from '@mui/material';
import { StandardizationOptions } from '../types';

interface StandardizationStepProps {
  options: StandardizationOptions;
  onUpdate: (options: StandardizationOptions) => void;
  onNext: () => void;
  onPrevious: () => void;
}

const StandardizationStep: React.FC<StandardizationStepProps> = ({ 
  options, 
  onUpdate, 
  onNext, 
  onPrevious 
}) => {
  // Handle currency change
  const handleCurrencyChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    onUpdate({
      ...options,
      currency: event.target.value as string
    });
  };

  // Handle category mapping change
  const handleCategoryMappingChange = (event: React.ChangeEvent<{ value: unknown }>) => {
    onUpdate({
      ...options,
      categoryMapping: event.target.value as string
    });
  };

  // Handle checkbox changes
  const handleCheckboxChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onUpdate({
      ...options,
      [event.target.name]: event.target.checked
    });
  };

  return (
    <Card>
      <Typography variant="h2" gutterBottom>
        Standardization Options
      </Typography>
      
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel id="currency-label">Currency</InputLabel>
            <Select
              labelId="currency-label"
              value={options.currency}
              onChange={(e) => handleCurrencyChange(e as any)}
              label="Currency"
            >
              <MenuItem value="USD">USD (US Dollar)</MenuItem>
              <MenuItem value="EUR">EUR (Euro)</MenuItem>
              <MenuItem value="GBP">GBP (British Pound)</MenuItem>
              <MenuItem value="CAD">CAD (Canadian Dollar)</MenuItem>
              <MenuItem value="AUD">AUD (Australian Dollar)</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12} md={6}>
          <FormControl fullWidth>
            <InputLabel id="category-mapping-label">Category Mapping</InputLabel>
            <Select
              labelId="category-mapping-label"
              value={options.categoryMapping}
              onChange={(e) => handleCategoryMappingChange(e as any)}
              label="Category Mapping"
            >
              <MenuItem value="Standard AV Categories">Standard AV Categories</MenuItem>
              <MenuItem value="Extended Categories">Extended Categories</MenuItem>
              <MenuItem value="Simple Categories">Simple Categories</MenuItem>
              <MenuItem value="Custom">Custom Mapping</MenuItem>
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12}>
          <FormControlLabel
            control={
              <Checkbox 
                checked={options.removeTax} 
                onChange={handleCheckboxChange}
                name="removeTax"
              />
            }
            label="Remove Tax from Prices"
          />
          
          <FormControlLabel
            control={
              <Checkbox 
                checked={options.includeCurrencySymbol} 
                onChange={handleCheckboxChange}
                name="includeCurrencySymbol"
              />
            }
            label="Include Currency Symbol"
          />
        </Grid>
      </Grid>
      
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
          Continue to Preview
        </Button>
      </Box>
    </Card>
  );
};

export default StandardizationStep;
