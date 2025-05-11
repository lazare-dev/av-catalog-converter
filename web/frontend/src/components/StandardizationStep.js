import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Card, 
  CardContent, 
  FormControl, 
  FormControlLabel, 
  FormGroup, 
  Switch, 
  Select, 
  MenuItem, 
  InputLabel, 
  Button, 
  Grid,
  Divider
} from '@mui/material';

const StandardizationStep = ({ options, onUpdate, onNext, onPrevious }) => {
  const [standardizationOptions, setStandardizationOptions] = useState(options);

  const handleCurrencyChange = (event) => {
    const updatedOptions = {
      ...standardizationOptions,
      currency: event.target.value
    };
    setStandardizationOptions(updatedOptions);
    onUpdate(updatedOptions);
  };

  const handleCategoryMappingChange = (event) => {
    const updatedOptions = {
      ...standardizationOptions,
      categoryMapping: event.target.value
    };
    setStandardizationOptions(updatedOptions);
    onUpdate(updatedOptions);
  };

  const handleSwitchChange = (event) => {
    const updatedOptions = {
      ...standardizationOptions,
      [event.target.name]: event.target.checked
    };
    setStandardizationOptions(updatedOptions);
    onUpdate(updatedOptions);
  };

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        Standardization Options
      </Typography>
      <Typography variant="body1" paragraph>
        Configure how you want your catalog data to be standardized.
      </Typography>

      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Grid container spacing={4}>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Currency Settings
              </Typography>
              <FormControl fullWidth sx={{ mb: 3 }}>
                <InputLabel id="currency-label">Currency</InputLabel>
                <Select
                  labelId="currency-label"
                  id="currency-select"
                  value={standardizationOptions.currency}
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

              <FormGroup>
                <FormControlLabel
                  control={
                    <Switch
                      checked={standardizationOptions.includeCurrencySymbol}
                      onChange={handleSwitchChange}
                      name="includeCurrencySymbol"
                      color="primary"
                    />
                  }
                  label="Include currency symbol"
                />
                <FormControlLabel
                  control={
                    <Switch
                      checked={standardizationOptions.removeTax}
                      onChange={handleSwitchChange}
                      name="removeTax"
                      color="primary"
                    />
                  }
                  label="Remove tax from prices"
                />
              </FormGroup>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography variant="h6" gutterBottom>
                Category Settings
              </Typography>
              <FormControl fullWidth sx={{ mb: 3 }}>
                <InputLabel id="category-mapping-label">Category Mapping</InputLabel>
                <Select
                  labelId="category-mapping-label"
                  id="category-mapping-select"
                  value={standardizationOptions.categoryMapping}
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
          </Grid>
        </CardContent>
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
          Continue to Preview
        </Button>
      </Box>
    </Box>
  );
};

export default StandardizationStep;
