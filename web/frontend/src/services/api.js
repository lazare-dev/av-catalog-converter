import axios from 'axios';

// API endpoints with proxy support
const API_PREFIX = '/api';

// Create axios instance with default config
const apiClient = axios.create({
  headers: {
    'Content-Type': 'application/json',
  },
});

// API endpoints
const endpoints = {
  health: `${API_PREFIX}/health`,
  upload: `${API_PREFIX}/upload`,
  analyze: `${API_PREFIX}/analyze`,
  mapFields: `${API_PREFIX}/map`,
  preview: `${API_PREFIX}/preview`,
  process: `${API_PREFIX}/process`,
  download: `${API_PREFIX}/download`,
  status: `${API_PREFIX}/status`,
};

// API service functions
export const apiService = {
  /**
   * Check API health
   */
  checkHealth: async () => {
    try {
      // Try to access the API status endpoint
      const response = await apiClient.get(endpoints.health);
      return {
        status: 'ok',
        message: response.data.message || 'API is operational',
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('Health check failed:', error);

      // Return error status but don't throw
      return {
        status: 'error',
        message: 'API is not responding',
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  },

  /**
   * Upload and analyze file
   */
  analyzeFile: async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      // First upload the file
      const uploadResponse = await axios.post(endpoints.upload, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      if (!uploadResponse.data.success) {
        throw new Error(uploadResponse.data.error || 'File upload failed');
      }

      const jobId = uploadResponse.data.job_id;

      // Then analyze the file
      const analyzeResponse = await apiClient.post(`${endpoints.analyze}`, {
        job_id: jobId
      });

      if (!analyzeResponse.data.success) {
        throw new Error(analyzeResponse.data.error || 'File analysis failed');
      }

      // Get structure info from the response
      const structure = analyzeResponse.data.structure_info || {};
      const columnInfo = analyzeResponse.data.column_info || {};

      // Get file info from job status
      const statusResponse = await apiClient.get(`${endpoints.status}/${jobId}`);

      return {
        fileInfo: {
          name: statusResponse.data.filename || file.name,
          type: structure.file_type || 'unknown',
          size: file.size / (1024 * 1024), // Convert to MB
          productCount: structure.row_count || 0,
        },
        structure,
        jobId
      };
    } catch (error) {
      console.error('File analysis failed:', error);
      throw error;
    }
  },

  /**
   * Get field mappings for a file
   */
  getFieldMappings: async (jobId) => {
    try {
      const response = await apiClient.get(`${endpoints.mapFields}/${jobId}`);

      if (!response.data.success) {
        throw new Error(response.data.error || 'Field mapping failed');
      }

      // Get field mappings from the response
      const fieldMappings = response.data.field_mappings || {};

      // Transform the response to match our frontend format
      const mappings = Object.entries(fieldMappings).map(([sourceColumn, mapping]) => ({
        sourceColumn,
        mapsTo: mapping.target_field || 'Do not include',
        confidence: Math.round((mapping.confidence || 0) * 100), // Convert 0-1 to 0-100
        sample: mapping.sample_value || ''
      }));

      return mappings;
    } catch (error) {
      console.error('Field mapping failed:', error);

      // If backend is not available, use fallback mock data for development
      if (process.env.NODE_ENV === 'development') {
        console.warn('Using fallback mock data for development');
        return [
          { sourceColumn: 'Item Name', mapsTo: 'Short Description', confidence: 98, sample: 'HD Camera 4K' },
          { sourceColumn: 'Item #', mapsTo: 'SKU', confidence: 95, sample: 'CAM-4K-01' },
          { sourceColumn: 'Cost (€)', mapsTo: 'Buy Cost', confidence: 90, sample: '€299.99' },
          { sourceColumn: 'Category', mapsTo: 'Category', confidence: 85, sample: 'Cameras' },
          { sourceColumn: 'Brand', mapsTo: 'Manufacturer', confidence: 92, sample: 'TechVision' },
          { sourceColumn: 'Model', mapsTo: 'Model', confidence: 88, sample: 'HDC-4000' },
          { sourceColumn: 'Description', mapsTo: 'Long Description', confidence: 94, sample: 'Professional 4K Ultra HD Camera with...' },
          { sourceColumn: 'Manufacturer Part #', mapsTo: 'Manufacturer SKU', confidence: 96, sample: 'TV-CAM-4K-01' }
        ];
      }

      throw error;
    }
  },

  /**
   * Update field mappings
   */
  updateFieldMappings: async (jobId, mappings) => {
    try {
      // Transform mappings to backend format
      const backendMappings = {};

      mappings.forEach(mapping => {
        backendMappings[mapping.sourceColumn] = {
          target_field: mapping.mapsTo === 'Do not include' ? null : mapping.mapsTo,
          confidence: mapping.confidence / 100, // Convert 0-100 to 0-1
          sample_value: mapping.sample
        };
      });

      const response = await apiClient.post(`${endpoints.mapFields}/${jobId}`, {
        mappings: backendMappings
      });

      if (!response.data.success) {
        throw new Error(response.data.error || 'Updating field mappings failed');
      }

      return response.data;
    } catch (error) {
      console.error('Updating field mappings failed:', error);
      throw error;
    }
  },

  /**
   * Get preview data based on mappings and options
   */
  getPreviewData: async (jobId, options) => {
    try {
      const response = await apiClient.get(`${endpoints.preview}/${jobId}`);

      if (!response.data.success) {
        throw new Error(response.data.error || 'Preview generation failed');
      }

      // Extract preview data
      const previewData = response.data.preview || [];

      // Get issues from job status
      const statusResponse = await apiClient.get(`${endpoints.status}/${jobId}`);

      // Extract and transform issues if available
      let issues = [];
      if (statusResponse.data.status === 'mapped' || statusResponse.data.status === 'completed') {
        // Get data quality issues from job
        const dataQualityIssues = statusResponse.data.data_quality_issues || [];

        // Transform issues to match our frontend format
        issues = dataQualityIssues.map(issue => ({
          type: issue.severity ? issue.severity.toLowerCase() : 'info', // 'ERROR', 'WARNING', 'INFO' -> 'error', 'warning', 'info'
          message: issue.message
        }));
      }

      return {
        previewData,
        issues
      };
    } catch (error) {
      console.error('Preview generation failed:', error);
      throw error;
    }
  },

  /**
   * Process catalog and get download URL
   */
  processCatalog: async (jobId, options) => {
    try {
      // First, process the file
      const processResponse = await apiClient.post(`${endpoints.process}/${jobId}`, {
        options: {
          currency: options.currency,
          category_mapping: options.categoryMapping,
          remove_tax: options.removeTax,
          include_currency_symbol: options.includeCurrencySymbol,
          output_format: 'csv'
        }
      });

      // Check if processing was successful
      if (!processResponse.data.success) {
        throw new Error(processResponse.data.error || 'Processing failed');
      }

      // Then get the download URL
      const downloadResponse = await apiClient.get(`${endpoints.download}/${jobId}`, {
        responseType: 'blob'
      });

      // Create a URL for the blob
      const url = window.URL.createObjectURL(new Blob([downloadResponse.data]));

      return url;
    } catch (error) {
      console.error('File processing failed:', error);
      throw error;
    }
  }
};

export default apiService;
