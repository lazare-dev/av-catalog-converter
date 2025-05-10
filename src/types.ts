// Application step types
export enum AppStep {
  UPLOAD = 'upload',
  FILE_INFO = 'fileInfo',
  FIELD_MAPPING = 'fieldMapping',
  STANDARDIZATION = 'standardization',
  PREVIEW = 'preview',
  ISSUES = 'issues',
  DOWNLOAD = 'download'
}

// File information
export interface FileInfo {
  name: string;
  type: string;
  size: number;
  productCount: number;
}

// Field mapping
export interface FieldMapping {
  sourceColumn: string;
  mapsTo: string;
  confidence: number;
  sample: string;
}

// Standardization options
export interface StandardizationOptions {
  currency: string;
  categoryMapping: string;
  removeTax: boolean;
  includeCurrencySymbol: boolean;
}

// Preview data
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
  WARNING = 'warning',
  ERROR = 'error',
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
