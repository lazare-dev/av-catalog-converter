import React from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Button, 
  FormControl,
  FormControlLabel,
  FormGroup,
  Switch,
  Select,
  MenuItem,
  InputLabel,
  Grid,
  SelectChangeEvent
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
  const handleCurrencyChange = (event: SelectChangeEvent) => {
    onUpdate({
      ...options,
      currency: event.target.value
    });
  };

  const handleCategoryMappingChange = (event: SelectChangeEvent) => {
    onUpdate({
      ...options,
      categoryMapping: event.target.value
    });
  };

  const handleSwitchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    onUpdate({
      ...options,
      [event.target.name]: event.target.checked
    });
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        Standardization Options
      </Typography>
      <Typography variant="body1" paragraph>
        Choose how you want to standardize your catalog data.
      </Typography>
      
      <Paper sx={{ p: 3, mb: 4 }}>
        <Grid container spacing={4}>
          <Grid item xs={12} md={6}>
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel id="currency-label">Currency</InputLabel>
              <Select
                labelId="currency-label"
                id="currency"
                value={options.currency}
                label="Currency"
                onChange={handleCurrencyChange}
              >
                <MenuItem value="USD">US Dollar (USD)</MenuItem>
                <MenuItem value="EUR">Euro (EUR)</MenuItem>
                <MenuItem value="GBP">British Pound (GBP)</MenuItem>
                <MenuItem value="CAD">Canadian Dollar (CAD)</MenuItem>
                <MenuItem value="AUD">Australian Dollar (AUD)</MenuItem>
              </Select>
            </FormControl>
            
            <FormControl fullWidth>
              <InputLabel id="category-mapping-label">Category Mapping</InputLabel>
              <Select
                labelId="category-mapping-label"
                id="category-mapping"
                value={options.categoryMapping}
                label="Category Mapping"
                onChange={handleCategoryMappingChange}
              >
                <MenuItem value="Standard AV Categories">Standard AV Categories</MenuItem>
                <MenuItem value="AVIXA Categories">AVIXA Categories</MenuItem>
                <MenuItem value="InfoComm Categories">InfoComm Categories</MenuItem>
                <MenuItem value="Custom Categories">Custom Categories</MenuItem>
              </Select>
            </FormControl>
          </Grid>
          
          <Grid item xs={12} md={6}>
            <FormGroup>
              <FormControlLabel
                control={
                  <Switch 
                    checked={options.removeTax} 
                    onChange={handleSwitchChange} 
                    name="removeTax" 
                  />
                }
                label="Remove tax from prices"
              />
              <Typography variant="body2" color="textSecondary" sx={{ ml: 4, mb: 2 }}>
                If prices include tax, this will convert them to pre-tax values.
              </Typography>
              
              <FormControlLabel
                control={
                  <Switch 
                    checked={options.includeCurrencySymbol} 
                    onChange={handleSwitchChange} 
                    name="includeCurrencySymbol" 
                  />
                }
                label="Include currency symbols in prices"
              />
              <Typography variant="body2" color="textSecondary" sx={{ ml: 4 }}>
                Add currency symbols to price values (e.g., $100.00 instead of 100.00).
              </Typography>
            </FormGroup>
          </Grid>
        </Grid>
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
        >
          Continue to Preview
        </Button>
      </Box>
    </Box>
  );
};

export default StandardizationStep;
