// Application step enum
export enum AppStep {
  UPLOAD = 'upload',
  FILE_INFO = 'file_info',
  FIELD_MAPPING = 'field_mapping',
  STANDARDIZATION = 'standardization',
  PREVIEW = 'preview',
  ISSUES = 'issues',
  DOWNLOAD = 'download'
}

// File information
export interface FileInfo {
  name: string;
  type: string;
  size: number; // in MB
  productCount: number;
}

// Field mapping
export interface FieldMapping {
  sourceColumn: string;
  mapsTo: string;
  confidence: number; // 0-100
  sample: string;
}

// Standardization options
export interface StandardizationOptions {
  currency: string;
  categoryMapping: string;
  removeTax: boolean;
  includeCurrencySymbol: boolean;
}

// Product data
export interface ProductData {
  sku: string;
  productName: string;
  price: string;
  category: string;
  manufacturer: string;
  description: string;
  specifications: string;
  edited?: boolean;
}

// Issue types
export enum IssueType {
  ERROR = 'error',
  WARNING = 'warning',
  INFO = 'info'
}

// Issue
export interface Issue {
  type: IssueType;
  message: string;
}

// Application state
export interface AppState {
  currentStep: AppStep;
  file: File | null;
  fileInfo: FileInfo | null;
  fieldMappings: FieldMapping[];
  standardizationOptions: StandardizationOptions;
  previewData: ProductData[];
  issues: Issue[];
  isProcessing: boolean;
  downloadUrl: string | null;
}
