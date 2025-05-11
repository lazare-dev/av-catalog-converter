import React from 'react';
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
  Button,
  Tooltip
} from '@mui/material';

const PreviewStep = ({ previewData, onNext, onPrevious }) => {
  // Define the exact columns that should be in the output CSV in this specific order
  const standardColumns = [
    'SKU',
    'Short Description',
    'Long Description',
    'Model',
    'Category Group',
    'Category',
    'Manufacturer',
    'Manufacturer SKU',
    'Image URL',
    'Document Name',
    'Document URL',
    'Unit Of Measure',
    'Buy Cost',
    'Trade Price',
    'MSRP GBP',
    'MSRP USD',
    'MSRP EUR',
    'Discontinued'
  ];

  // Get all unique keys from the preview data
  const dataKeys = previewData.length > 0
    ? Array.from(new Set(previewData.flatMap(item => Object.keys(item))))
    : [];

  // Filter out the 'edited' key which is just for UI purposes
  const dataColumns = dataKeys.filter(key => key !== 'edited');

  // Use standard columns, but add any additional columns from the data
  const columns = [...standardColumns, ...dataColumns.filter(col => !standardColumns.includes(col))];

  return (
    <Box sx={{ mt: 4 }}>
      <Typography variant="h2" gutterBottom>
        Data Preview
      </Typography>
      <Typography variant="body1" paragraph>
        Preview how your data will look after conversion. We're showing the first few rows.
      </Typography>

      <Card sx={{ mb: 4, overflow: 'hidden' }}>
        <TableContainer component={Paper} sx={{ maxHeight: 400 }}>
          <Table stickyHeader>
            <TableHead>
              <TableRow>
                {columns.map((column) => (
                  <TableCell
                    key={column}
                    sx={{
                      fontWeight: 'bold',
                      backgroundColor: '#f5f5f5'
                    }}
                  >
                    {column.charAt(0).toUpperCase() + column.slice(1)}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              {previewData.map((row, rowIndex) => (
                <TableRow
                  key={rowIndex}
                  sx={{
                    backgroundColor: row.edited ? 'rgba(46, 204, 113, 0.1)' : 'inherit',
                    '&:hover': { backgroundColor: 'rgba(52, 152, 219, 0.1)' }
                  }}
                >
                  {columns.map((column) => (
                    <TableCell key={`${rowIndex}-${column}`}>
                      {row[column] !== undefined && row[column] !== '' ? row[column] : (
                        <span style={{ color: '#ccc', fontStyle: 'italic' }}>Empty</span>
                      )}
                    </TableCell>
                  ))}
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
          Continue to Issues
        </Button>
      </Box>
    </Box>
  );
};

export default PreviewStep;
