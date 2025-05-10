import axios from 'axios';
import { FieldMapping, ProductData, Issue, StandardizationOptions } from '../types';

// Base API URL - change this to your backend URL
const API_BASE_URL = 'http://localhost:8080/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API endpoints
const endpoints = {
  health: '/health',
  upload: '/upload',
  analyze: '/analyze',
  mapFields: '/map-fields',
};

// API service functions
export const apiService = {
  /**
   * Check API health
   */
  checkHealth: async () => {
    try {
      const response = await apiClient.get(endpoints.health);
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  /**
   * Upload a file for analysis
   * @param file The file to upload
   */
  analyzeFile: async (file: File): Promise<{
    fileInfo: {
      name: string;
      type: string;
      size: number;
      productCount: number;
    };
    structure: any;
  }> => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(`${API_BASE_URL}${endpoints.analyze}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Extract file info from response
      const { file_info, structure } = response.data;
      
      return {
        fileInfo: {
          name: file_info.filename,
          type: file_info.parser,
          size: file_info.size / (1024 * 1024), // Convert to MB
          productCount: structure.column_count || 0,
        },
        structure,
      };
    } catch (error) {
      console.error('File analysis failed:', error);
      throw error;
    }
  },

  /**
   * Get field mappings for the uploaded file
   * @param columns Column names from the file
   * @param sampleData Sample data from the file
   */
  getFieldMappings: async (
    columns: string[],
    sampleData: any[]
  ): Promise<FieldMapping[]> => {
    try {
      const response = await apiClient.post(endpoints.mapFields, {
        columns,
        sample_data: sampleData,
      });

      // Convert API response to FieldMapping array
      const mappings = response.data.mappings.map((mapping: any) => ({
        sourceColumn: mapping.source_column,
        mapsTo: mapping.maps_to,
        confidence: mapping.confidence * 100, // Convert from 0-1 to 0-100
        sample: mapping.sample_value,
      }));

      return mappings;
    } catch (error) {
      console.error('Field mapping failed:', error);
      throw error;
    }
  },

  /**
   * Process the file with the given mappings and options
   * @param file The file to process
   * @param mappings Field mappings
   * @param options Standardization options
   */
  processFile: async (
    file: File,
    mappings: FieldMapping[],
    options: StandardizationOptions
  ): Promise<string> => {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('format', 'csv'); // Default to CSV format
      
      // Add mappings and options as JSON
      formData.append('mappings', JSON.stringify(
        mappings.map(m => ({
          source_column: m.sourceColumn,
          target_field: m.mapsTo === 'Do not include' ? null : m.mapsTo,
        }))
      ));
      
      formData.append('options', JSON.stringify({
        currency: options.currency,
        category_mapping: options.categoryMapping,
        remove_tax: options.removeTax,
        include_currency_symbol: options.includeCurrencySymbol,
      }));

      const response = await axios.post(`${API_BASE_URL}${endpoints.upload}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob', // Important for file download
      });

      // Create a URL for the blob
      const url = window.URL.createObjectURL(new Blob([response.data]));
      
      return url;
    } catch (error) {
      console.error('File processing failed:', error);
      throw error;
    }
  },

  /**
   * Get preview data based on mappings and options
   * This would typically be a separate endpoint in a real API
   * For now, we'll simulate it by processing a small sample
   */
  getPreviewData: async (
    file: File,
    mappings: FieldMapping[],
    options: StandardizationOptions
  ): Promise<{
    previewData: ProductData[];
    issues: Issue[];
  }> => {
    try {
      // In a real implementation, you would have a dedicated preview endpoint
      // For now, we'll use the analyze endpoint and transform the data
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post(`${API_BASE_URL}${endpoints.analyze}`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      // Extract sample data from response
      const sampleData = response.data.sample_data || [];
      
      // Transform sample data based on mappings
      const previewData: ProductData[] = sampleData.map((row: any, index: number) => {
        const transformedRow: any = {};
        
        // Apply mappings
        mappings.forEach(mapping => {
          if (mapping.mapsTo !== 'Do not include') {
            const fieldKey = mapping.mapsTo.toLowerCase().replace(/\s+/g, '');
            transformedRow[fieldKey] = row[mapping.sourceColumn] || '';
          }
        });
        
        // Create a ProductData object
        return {
          sku: transformedRow.sku || `ITEM-${index + 1}`,
          productName: transformedRow.productname || 'Unknown Product',
          price: transformedRow.price ? 
            (options.includeCurrencySymbol ? `$${transformedRow.price}` : transformedRow.price) : 
            '',
          category: transformedRow.productcategory || '',
          manufacturer: transformedRow.manufacturer || '',
          description: transformedRow.description || '',
          specifications: transformedRow.specifications || '',
        };
      });
      
      // Generate sample issues
      const issues = [];
      
      // Check for missing values
      const missingSpecs = previewData.filter(p => !p.specifications).length;
      if (missingSpecs > 0) {
        issues.push({
          type: 'warning',
          message: `${missingSpecs} products have missing specifications`
        });
      }
      
      // Check for duplicate SKUs
      const skus = previewData.map(p => p.sku);
      const uniqueSkus = new Set(skus);
      if (skus.length !== uniqueSkus.size) {
        issues.push({
          type: 'error',
          message: `${skus.length - uniqueSkus.size} duplicate SKUs detected`
        });
      }
      
      // Add currency conversion info if applicable
      if (options.currency !== 'USD') {
        issues.push({
          type: 'info',
          message: `Prices were converted from ${options.currency} to USD`
        });
      }
      
      return {
        previewData,
        issues,
      };
    } catch (error) {
      console.error('Preview generation failed:', error);
      throw error;
    }
  },
};
