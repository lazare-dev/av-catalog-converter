import axios from 'axios';
import { FieldMapping, ProductData, Issue, IssueType, StandardizationOptions } from '../types';
import { logger } from './logging';

// Base API URL - change this to your backend URL
const API_BASE_URL = 'http://localhost:8080/api';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    logger.info('API', `Request: ${config.method?.toUpperCase()} ${config.url}`, {
      headers: config.headers,
      params: config.params,
      data: config.data
    });
    return config;
  },
  (error) => {
    logger.error('API', 'Request error', { error });
    return Promise.reject(error);
  }
);

// Add response interceptor for logging
apiClient.interceptors.response.use(
  (response) => {
    logger.info('API', `Response: ${response.status} ${response.statusText}`, {
      url: response.config.url,
      data: response.data ? 'Data received' : 'No data',
      size: response.headers['content-length'] || 'unknown'
    });
    return response;
  },
  (error) => {
    logger.error('API', 'Response error', {
      url: error.config?.url,
      status: error.response?.status,
      statusText: error.response?.statusText,
      message: error.message
    });
    return Promise.reject(error);
  }
);

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

      const response = await apiClient.post(endpoints.analyze, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Log the response structure for debugging
      logger.debug('API', 'Analyze file response structure', {
        keys: Object.keys(response.data),
        hasFileInfo: 'file_info' in response.data,
        hasStructure: 'structure' in response.data,
        responseData: JSON.stringify(response.data)
      });

      // Check if the response indicates an error
      if (response.data.error) {
        logger.error('API', 'Server returned an error during file analysis', {
          error: response.data.error,
          details: response.data.details || 'No details provided'
        });
        throw new Error(response.data.error);
      }

      // Extract file info from response - handle both response formats
      // The backend returns file_info in the response
      // Make sure response.data exists and is an object
      const responseData = response.data && typeof response.data === 'object' ? response.data : {};

      // Log the actual file_info structure if it exists
      if (responseData.file_info) {
        logger.debug('API', 'File info structure', {
          fileInfo: JSON.stringify(responseData.file_info)
        });
      } else {
        logger.warning('API', 'No file_info in response', {
          availableKeys: Object.keys(responseData)
        });
      }

      // Make sure fileInfo is always at least an empty object
      const fileInfo = responseData.file_info && typeof responseData.file_info === 'object'
                      ? responseData.file_info
                      : {};

      const structure = responseData.structure && typeof responseData.structure === 'object'
                      ? responseData.structure
                      : {};

      // Calculate product count from structure or file_info
      const productCount = (fileInfo && typeof fileInfo === 'object' && 'product_count' in fileInfo)
                          ? fileInfo.product_count
                          : (structure && typeof structure === 'object' && 'column_count' in structure
                             ? structure.column_count
                             : 0);

      // Ensure we have a valid filename - check that fileInfo exists and has filename property
      const filename = (fileInfo && typeof fileInfo === 'object' && 'filename' in fileInfo)
                      ? fileInfo.filename
                      : file.name;

      // Create a safe fileInfo object with all required properties
      const safeFileInfo = {
        name: filename,
        type: (fileInfo && typeof fileInfo === 'object' && 'parser' in fileInfo)
              ? fileInfo.parser
              : file.type,
        size: (fileInfo && typeof fileInfo === 'object' && 'size' in fileInfo)
              ? fileInfo.size / (1024 * 1024)
              : file.size / (1024 * 1024), // Convert to MB
        productCount: productCount,
      };

      // Log the safe fileInfo object we're returning
      logger.debug('API', 'Returning fileInfo', {
        safeFileInfo: JSON.stringify(safeFileInfo)
      });

      return {
        fileInfo: safeFileInfo,
        structure: structure || {},
      };
    } catch (error: any) {
      // Make sure error is an object with a message property
      const errorMessage = error && typeof error === 'object' && error.message
                          ? error.message
                          : 'Unknown error';

      console.error('File analysis failed:', error || 'Unknown error');

      // Rethrow with a more descriptive message
      throw new Error(`File analysis failed: ${errorMessage}`);
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
    } catch (error: any) {
      // Make sure error is an object with a message property
      const errorMessage = error && typeof error === 'object' && error.message
                          ? error.message
                          : 'Unknown error';

      console.error('Field mapping failed:', error || 'Unknown error');

      // Rethrow with a more descriptive message
      throw new Error(`Field mapping failed: ${errorMessage}`);
    }
  },

  /**
   * Process the file with the given mappings and options
   * @param file The file to process
   * @param mappings Field mappings
   * @param options Standardization options
   */
  processFileWithMappings: async (
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

      const response = await apiClient.post(endpoints.upload, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob', // Important for file download
      });

      // Create a URL for the blob
      const url = window.URL.createObjectURL(new Blob([response.data]));

      return url;
    } catch (error: any) {
      console.error('File processing failed:', error);
      throw error;
    }
  },

  /**
   * Process a file with simplified options (for the simple frontend)
   * @param file The file to process
   * @param format Output format (csv, excel, json)
   */
  processFile: async (
    file: File,
    format: string = 'csv'
  ): Promise<string> => {
    try {
      logger.info('API', 'Processing file with simplified options', {
        fileName: file.name,
        fileSize: file.size,
        format
      });

      const formData = new FormData();
      formData.append('file', file);
      formData.append('format', format);

      const response = await apiClient.post(endpoints.upload, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        responseType: 'blob', // Important for file download
      });

      // Check if the response is an error (sometimes errors come back as JSON even with responseType: 'blob')
      if (response.headers['content-type']?.includes('application/json')) {
        // Try to parse the blob as JSON to get the error message
        try {
          const errorText = await response.data.text();
          const errorJson = JSON.parse(errorText);
          if (errorJson.error) {
            logger.error('API', 'Server returned an error during file processing', {
              error: errorJson.error,
              details: errorJson.details || 'No details provided'
            });
            throw new Error(errorJson.error);
          }
        } catch (parseError) {
          // If we can't parse as JSON, just continue with blob handling
          logger.warning('API', 'Could not parse response as JSON', { parseError });
        }
      }

      // Create a URL for the blob
      const url = window.URL.createObjectURL(new Blob([response.data]));
      logger.info('API', 'File processed successfully', { format });

      return url;
    } catch (error: any) {
      // Make sure error is an object with message and stack properties
      const errorMessage = error && typeof error === 'object' && error.message
                          ? error.message
                          : 'Unknown error';

      const errorStack = error && typeof error === 'object' && error.stack
                        ? error.stack
                        : 'Stack trace not available';

      logger.error('API', 'File processing failed', {
        error: errorMessage,
        stack: errorStack
      });

      // Rethrow with a more descriptive message
      throw new Error(`File processing failed: ${errorMessage}`);
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

      const response = await apiClient.post(endpoints.analyze, formData, {
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
          type: IssueType.WARNING,
          message: `${missingSpecs} products have missing specifications`
        });
      }

      // Check for duplicate SKUs
      const skus = previewData.map(p => p.sku);
      const uniqueSkus = new Set(skus);
      if (skus.length !== uniqueSkus.size) {
        issues.push({
          type: IssueType.ERROR,
          message: `${skus.length - uniqueSkus.size} duplicate SKUs detected`
        });
      }

      // Add currency conversion info if applicable
      if (options.currency !== 'USD') {
        issues.push({
          type: IssueType.INFO,
          message: `Prices were converted from ${options.currency} to USD`
        });
      }

      return {
        previewData,
        issues,
      };
    } catch (error: any) {
      // Make sure error is an object with a message property
      const errorMessage = error && typeof error === 'object' && error.message
                          ? error.message
                          : 'Unknown error';

      console.error('Preview generation failed:', error || 'Unknown error');

      // Rethrow with a more descriptive message
      throw new Error(`Preview generation failed: ${errorMessage}`);
    }
  },
};
