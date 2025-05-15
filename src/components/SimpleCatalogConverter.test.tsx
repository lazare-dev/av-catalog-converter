import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';
import SimpleCatalogConverter from './SimpleCatalogConverter';
import { apiService } from '../services/api';
import { logger } from '../services/logging';

// Mock the API service
jest.mock('../services/api', () => ({
  apiService: {
    analyzeFile: jest.fn(),
    processFile: jest.fn()
  }
}));

// Mock the logger
jest.mock('../services/logging', () => ({
  logger: {
    debug: jest.fn(),
    info: jest.fn(),
    warning: jest.fn(),
    error: jest.fn(),
    critical: jest.fn()
  }
}));

describe('SimpleCatalogConverter Component', () => {
  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
  });

  test('renders the component with initial state', () => {
    render(<SimpleCatalogConverter />);
    
    // Check that the title is rendered
    expect(screen.getByText('AV Catalog Converter')).toBeInTheDocument();
    
    // Check that the upload button is rendered
    expect(screen.getByText('Select Catalog File')).toBeInTheDocument();
    
    // Check that the stepper is rendered with the correct steps
    expect(screen.getByText('Upload File')).toBeInTheDocument();
    expect(screen.getByText('Validate')).toBeInTheDocument();
    expect(screen.getByText('Export')).toBeInTheDocument();
    
    // Verify logger was called
    expect(logger.info).toHaveBeenCalledWith('SimpleCatalogConverter', 'Component mounted');
  });

  test('handles file selection', async () => {
    render(<SimpleCatalogConverter />);
    
    // Create a mock file
    const file = new File(['test content'], 'test.csv', { type: 'text/csv' });
    
    // Get the hidden file input
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    expect(fileInput).not.toBeNull();
    
    // Simulate file selection
    userEvent.upload(fileInput, file);
    
    // Check that the file name is displayed
    await waitFor(() => {
      expect(screen.getByText(/test.csv/)).toBeInTheDocument();
    });
    
    // Check that the output format selector is displayed
    expect(screen.getByLabelText('Output Format')).toBeInTheDocument();
    
    // Check that the validate button is displayed
    expect(screen.getByText('Validate File')).toBeInTheDocument();
    
    // Verify logger was called with file info
    expect(logger.info).toHaveBeenCalledWith(
      'SimpleCatalogConverter', 
      'File selected', 
      expect.objectContaining({
        fileName: 'test.csv'
      })
    );
  });

  test('validates file successfully', async () => {
    // Mock successful file analysis
    (apiService.analyzeFile as jest.Mock).mockResolvedValue({
      fileInfo: {
        name: 'test.csv',
        type: 'CSV',
        size: 0.5, // 0.5 MB
        productCount: 10
      },
      structure: {
        columns: ['sku', 'name', 'price', 'description']
      }
    });
    
    render(<SimpleCatalogConverter />);
    
    // Create a mock file
    const file = new File(['test content'], 'test.csv', { type: 'text/csv' });
    
    // Get the hidden file input
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    
    // Simulate file selection
    userEvent.upload(fileInput, file);
    
    // Click the validate button
    await waitFor(() => {
      const validateButton = screen.getByText('Validate File');
      fireEvent.click(validateButton);
    });
    
    // Check that the API was called
    expect(apiService.analyzeFile).toHaveBeenCalledWith(file);
    
    // Check that we moved to the validation step
    await waitFor(() => {
      expect(screen.getByText('Validation Results')).toBeInTheDocument();
    });
    
    // Check that file info is displayed
    expect(screen.getByText(/Name: test.csv/)).toBeInTheDocument();
    expect(screen.getByText(/Products: 10/)).toBeInTheDocument();
    
    // Check that the process button is enabled
    const processButton = screen.getByText('Process & Export');
    expect(processButton).not.toBeDisabled();
    
    // Verify logger was called
    expect(logger.info).toHaveBeenCalledWith(
      'SimpleCatalogConverter', 
      'File validation complete', 
      expect.anything()
    );
  });

  test('handles validation errors', async () => {
    // Mock file analysis with no products
    (apiService.analyzeFile as jest.Mock).mockResolvedValue({
      fileInfo: {
        name: 'empty.csv',
        type: 'CSV',
        size: 0.1, // 0.1 MB
        productCount: 0
      },
      structure: {
        columns: ['column1', 'column2']
      }
    });
    
    render(<SimpleCatalogConverter />);
    
    // Create a mock file
    const file = new File(['empty content'], 'empty.csv', { type: 'text/csv' });
    
    // Get the hidden file input
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    
    // Simulate file selection
    userEvent.upload(fileInput, file);
    
    // Click the validate button
    await waitFor(() => {
      const validateButton = screen.getByText('Validate File');
      fireEvent.click(validateButton);
    });
    
    // Check that the API was called
    expect(apiService.analyzeFile).toHaveBeenCalledWith(file);
    
    // Check that validation error is displayed
    await waitFor(() => {
      expect(screen.getByText('No products detected in the file')).toBeInTheDocument();
    });
    
    // Verify logger was called with error
    expect(logger.info).toHaveBeenCalledWith(
      'SimpleCatalogConverter', 
      'File analysis complete', 
      expect.anything()
    );
  });

  test('processes and exports file', async () => {
    // Mock successful file analysis
    (apiService.analyzeFile as jest.Mock).mockResolvedValue({
      fileInfo: {
        name: 'test.csv',
        type: 'CSV',
        size: 0.5,
        productCount: 10
      },
      structure: {
        columns: ['sku', 'name', 'price', 'description']
      }
    });
    
    // Mock successful file processing
    const mockDownloadUrl = 'blob:http://localhost/mock-url';
    (apiService.processFile as jest.Mock).mockResolvedValue(mockDownloadUrl);
    
    render(<SimpleCatalogConverter />);
    
    // Create a mock file
    const file = new File(['test content'], 'test.csv', { type: 'text/csv' });
    
    // Get the hidden file input
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    
    // Simulate file selection
    userEvent.upload(fileInput, file);
    
    // Click the validate button
    await waitFor(() => {
      const validateButton = screen.getByText('Validate File');
      fireEvent.click(validateButton);
    });
    
    // Click the process button
    await waitFor(() => {
      const processButton = screen.getByText('Process & Export');
      fireEvent.click(processButton);
    });
    
    // Check that the API was called
    expect(apiService.processFile).toHaveBeenCalledWith(file, 'csv');
    
    // Check that we moved to the export step
    await waitFor(() => {
      expect(screen.getByText('Processing Complete!')).toBeInTheDocument();
    });
    
    // Check that the download button is displayed
    expect(screen.getByText('Download Converted File')).toBeInTheDocument();
    
    // Verify logger was called
    expect(logger.info).toHaveBeenCalledWith(
      'SimpleCatalogConverter', 
      'File processing complete', 
      expect.objectContaining({
        downloadUrl: mockDownloadUrl
      })
    );
  });

  test('handles API errors during processing', async () => {
    // Mock successful file analysis
    (apiService.analyzeFile as jest.Mock).mockResolvedValue({
      fileInfo: {
        name: 'test.csv',
        type: 'CSV',
        size: 0.5,
        productCount: 10
      },
      structure: {
        columns: ['sku', 'name', 'price', 'description']
      }
    });
    
    // Mock API error during processing
    (apiService.processFile as jest.Mock).mockRejectedValue(new Error('Processing failed'));
    
    render(<SimpleCatalogConverter />);
    
    // Create a mock file
    const file = new File(['test content'], 'test.csv', { type: 'text/csv' });
    
    // Get the hidden file input
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
    
    // Simulate file selection
    userEvent.upload(fileInput, file);
    
    // Click the validate button
    await waitFor(() => {
      const validateButton = screen.getByText('Validate File');
      fireEvent.click(validateButton);
    });
    
    // Click the process button
    await waitFor(() => {
      const processButton = screen.getByText('Process & Export');
      fireEvent.click(processButton);
    });
    
    // Check that the API was called
    expect(apiService.processFile).toHaveBeenCalledWith(file, 'csv');
    
    // Check that error snackbar is displayed
    await waitFor(() => {
      expect(screen.getByText('Failed to process file')).toBeInTheDocument();
    });
    
    // Verify logger was called with error
    expect(logger.error).toHaveBeenCalledWith(
      'SimpleCatalogConverter', 
      'File processing failed', 
      expect.objectContaining({
        error: expect.anything()
      })
    );
  });
});
