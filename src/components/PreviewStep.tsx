import React from 'react';
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
  Link,
  CircularProgress
} from '@mui/material';
import { ProductData } from '../types';

interface PreviewStepProps {
  previewData: ProductData[];
  onNext: () => void;
  onPrevious: () => void;
  isLoading?: boolean;
}

const PreviewStep: React.FC<PreviewStepProps> = ({
  previewData,
  onNext,
  onPrevious,
  isLoading = false
}) => {
  return (
    <Card>
      <Typography variant="h2" gutterBottom>
        Preview
      </Typography>

      {isLoading ? (
        <Box sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          p: 5,
          minHeight: 300
        }}>
          <CircularProgress size={40} sx={{ mb: 2 }} />
          <Typography variant="body1">
            Generating preview...
          </Typography>
        </Box>
      ) : (
        <>
          <TableContainer sx={{ mb: 2 }}>
            <Table>
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
                  <TableRow key={index}>
                    <TableCell>{product.sku}</TableCell>
                    <TableCell>{product.productName}</TableCell>
                    <TableCell>{product.price}</TableCell>
                    <TableCell sx={product.edited ? { position: 'relative', '&::after': { content: '"*"', color: 'primary.main', fontWeight: 'bold' } } : {}}>
                      {product.category}
                    </TableCell>
                    <TableCell>{product.manufacturer}</TableCell>
                    <TableCell>{product.description}</TableCell>
                    <TableCell>{product.specifications}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>

          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="body2" color="textSecondary">
              * Edited value
            </Typography>
            <Box>
              <Link href="#" sx={{ ml: 2 }}>Show All ({previewData.length})</Link>
              <Link href="#" sx={{ ml: 2 }}>Edit</Link>
            </Box>
          </Box>
        </>
      )}

      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Button
          variant="outlined"
          onClick={onPrevious}
          disabled={isLoading}
        >
          Back
        </Button>
        <Button
          variant="contained"
          onClick={onNext}
          disabled={isLoading}
        >
          Continue to Issues
        </Button>
      </Box>
    </Card>
  );
};

export default PreviewStep;
