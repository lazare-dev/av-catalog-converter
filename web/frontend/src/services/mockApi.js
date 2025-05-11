// Mock API service for development without a backend

// Sample data for demonstration
const sampleFieldMappings = [
  { sourceColumn: 'Item Name', mapsTo: 'Product Name', confidence: 98, sample: 'HD Camera 4K' },
  { sourceColumn: 'Item #', mapsTo: 'SKU', confidence: 95, sample: 'CAM-4K-01' },
  { sourceColumn: 'Cost (€)', mapsTo: 'Price', confidence: 90, sample: '€299.99' },
  { sourceColumn: 'Category', mapsTo: 'Product Category', confidence: 85, sample: 'Cameras' },
  { sourceColumn: 'Brand', mapsTo: 'Manufacturer', confidence: 92, sample: 'TechVision' },
  { sourceColumn: 'Stock', mapsTo: 'Do not include', confidence: 0, sample: 'In Stock' }
];

// Define all possible columns for the preview data
const allPossibleColumns = [
  'sku',
  'productName',
  'price',
  'category',
  'manufacturer',
  'description',
  'specifications',
  'weight',
  'dimensions',
  'warranty',
  'availability',
  'rating'
];

// Create sample preview data with some empty columns
const samplePreviewData = [
  {
    sku: 'CAM-4K-01',
    productName: 'HD Camera 4K',
    price: '$325.49',
    category: 'Cameras',
    manufacturer: 'TechVision',
    description: '4K Ultra HD...',
    specifications: 'Resolution: 4K',
    weight: '1.2 kg',
    dimensions: '10 x 8 x 6 cm',
    warranty: '2 years',
    availability: '',
    rating: '4.5/5'
  },
  {
    sku: 'MIC-WL-02',
    productName: 'Wireless Mic',
    price: '$162.75',
    category: 'Microphones',
    manufacturer: 'AudioPro',
    description: 'Professional...',
    specifications: 'Frequency: 2.4GHz',
    weight: '0.3 kg',
    dimensions: '',
    warranty: '1 year',
    availability: 'In Stock',
    rating: '',
    edited: true
  },
  {
    sku: 'SPK-BT-03',
    productName: 'Bluetooth Speaker',
    price: '$108.50',
    category: 'Speakers',
    manufacturer: 'SoundMax',
    description: 'Portable BT...',
    specifications: 'Power: 30W',
    weight: '0.8 kg',
    dimensions: '15 x 8 x 8 cm',
    warranty: '',
    availability: 'Out of Stock',
    rating: '3.8/5'
  }
];

const sampleIssues = [
  { type: 'warning', message: '3 products have missing specifications' },
  { type: 'error', message: '1 duplicate SKU detected' },
  { type: 'info', message: '5 prices were converted from EUR to USD' }
];

// Mock API service
const mockApiService = {
  /**
   * Check API health
   */
  checkHealth: async () => {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          status: 'ok',
          version: '1.0.0',
          app_name: 'AV Catalog Converter'
        });
      }, 500);
    });
  },

  /**
   * Analyze uploaded file
   */
  analyzeFile: async (file) => {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          fileInfo: {
            name: file.name,
            type: file.type.includes('sheet') ? 'Excel Spreadsheet' : 'CSV File',
            size: file.size / (1024 * 1024), // Convert to MB
            productCount: 245 // This would come from the backend in a real app
          },
          structure: {
            column_count: 12,
            row_count: 245,
            has_headers: true
          }
        });
      }, 1000);
    });
  },

  /**
   * Get field mappings
   */
  getFieldMappings: async () => {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(sampleFieldMappings);
      }, 800);
    });
  },

  /**
   * Get preview data based on mappings and options
   */
  getPreviewData: async (file, mappings, options) => {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          previewData: samplePreviewData,
          issues: sampleIssues
        });
      }, 1200);
    });
  },

  /**
   * Process catalog and get download URL
   */
  processCatalog: async (file, mappings, options) => {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve('processed-catalog.csv');
      }, 2000);
    });
  }
};

export default mockApiService;
