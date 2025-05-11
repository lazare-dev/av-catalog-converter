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
  Chip
} from '@mui/material';
import { ProductData } from '../types';

interface PreviewStepProps {
  previewData: ProductData[];
  onNext: () => void;
  onPrevious: () => void;
}

const PreviewStep: React.FC<PreviewStepProps> = ({ 
  previewData, 
  onNext, 
  onPrevious 
}) => {
  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        Data Preview
      </Typography>
      <Typography variant="body1" paragraph>
        Preview how your data will look after standardization. This shows a sample of the first few products.
      </Typography>
      
      <TableContainer component={Paper} sx={{ mb: 4, maxHeight: 400, overflow: 'auto' }}>
        <Table stickyHeader>
          <TableHead>
            <TableRow>
              <TableCell>SKU</TableCell>
              <TableCell>Product Name</TableCell>
              <TableCell>Price</TableCell>
              <TableCell>Category</TableCell>
              <TableCell>Manufacturer</TableCell>
              <TableCell>Description</TableCell>
              <TableCell>Specifications</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {previewData.map((product, index) => (
              <TableRow 
                key={index}
                sx={{ 
                  backgroundColor: product.edited ? 'rgba(46, 204, 113, 0.1)' : 'inherit'
                }}
              >
                <TableCell>{product.sku}</TableCell>
                <TableCell>{product.productName}</TableCell>
                <TableCell>{product.price}</TableCell>
                <TableCell>{product.category}</TableCell>
                <TableCell>{product.manufacturer}</TableCell>
                <TableCell>{product.description}</TableCell>
                <TableCell>{product.specifications}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
      
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 4 }}>
        <Chip 
          label="Edited values" 
          sx={{ 
            backgroundColor: 'rgba(46, 204, 113, 0.1)', 
            borderColor: 'rgba(46, 204, 113, 0.5)',
            borderWidth: 1,
            borderStyle: 'solid',
            mr: 2
          }} 
        />
        <Typography variant="body2" color="textSecondary">
          Highlighted rows contain values that have been standardized or modified
        </Typography>
      </Box>
      
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
          Continue to Issues
        </Button>
      </Box>
    </Box>
  );
};

export default PreviewStep;
