import React, { useState } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { Container, Box, Typography, CssBaseline } from '@mui/material';
import theme from './theme';
import UploadStep from './components/UploadStep';
import FileInfoStep from './components/FileInfoStep';
import FieldMappingStep from './components/FieldMappingStep';
import StandardizationStep from './components/StandardizationStep';
import PreviewStep from './components/PreviewStep';
import IssuesStep from './components/IssuesStep';
import DownloadStep from './components/DownloadStep';
import {
  AppStep,
  AppState,
  FileInfo,
  FieldMapping,
  StandardizationOptions,
  ProductData,
  Issue,
  IssueType
} from './types';

// Initial state
const initialState: AppState = {
  currentStep: AppStep.UPLOAD,
  file: null,
  fileInfo: null,
  fieldMappings: [],
  standardizationOptions: {
    currency: 'USD',
    categoryMapping: 'Standard AV Categories',
    removeTax: true,
    includeCurrencySymbol: true
  },
  previewData: [],
  issues: [],
  isProcessing: false,
  downloadUrl: null
};

// Sample data for demonstration
const sampleFieldMappings: FieldMapping[] = [
  { sourceColumn: 'Item Name', mapsTo: 'Product Name', confidence: 98, sample: 'HD Camera 4K' },
  { sourceColumn: 'Item #', mapsTo: 'SKU', confidence: 95, sample: 'CAM-4K-01' },
  { sourceColumn: 'Cost (€)', mapsTo: 'Price', confidence: 90, sample: '€299.99' },
  { sourceColumn: 'Category', mapsTo: 'Product Category', confidence: 85, sample: 'Cameras' },
  { sourceColumn: 'Brand', mapsTo: 'Manufacturer', confidence: 92, sample: 'TechVision' },
  { sourceColumn: 'Stock', mapsTo: 'Do not include', confidence: 0, sample: 'In Stock' }
];

const samplePreviewData: ProductData[] = [
  {
    sku: 'CAM-4K-01',
    productName: 'HD Camera 4K',
    price: '$325.49',
    category: 'Cameras',
    manufacturer: 'TechVision',
    description: '4K Ultra HD...',
    specifications: 'Resolution: 4K'
  },
  {
    sku: 'MIC-WL-02',
    productName: 'Wireless Mic',
    price: '$162.75',
    category: 'Microphones',
    manufacturer: 'AudioPro',
    description: 'Professional...',
    specifications: 'Frequency: 2.4GHz',
    edited: true
  },
  {
    sku: 'SPK-BT-03',
    productName: 'Bluetooth Speaker',
    price: '$108.50',
    category: 'Speakers',
    manufacturer: 'SoundMax',
    description: 'Portable BT...',
    specifications: 'Power: 30W'
  }
];

const sampleIssues: Issue[] = [
  { type: IssueType.WARNING, message: '3 products have missing specifications' },
  { type: IssueType.ERROR, message: '1 duplicate SKU detected' },
  { type: IssueType.INFO, message: '5 prices were converted from EUR to USD' }
];

function App() {
  const [state, setState] = useState<AppState>(initialState);

  // Handle file upload
  const handleFileUpload = (file: File) => {
    setState({
      ...state,
      file,
      currentStep: AppStep.FILE_INFO,
      fileInfo: {
        name: file.name,
        type: file.type.includes('sheet') ? 'Excel Spreadsheet' : 'CSV File',
        size: file.size / (1024 * 1024), // Convert to MB
        productCount: 245 // This would come from the backend in a real app
      }
    });

    // In a real app, you would upload the file to the server here
    // and get back the file info, field mappings, etc.

    // For demo purposes, we'll simulate this with a timeout
    setTimeout(() => {
      setState(prevState => ({
        ...prevState,
        fieldMappings: sampleFieldMappings
      }));
    }, 1000);
  };

  // Handle next step
  const handleNextStep = () => {
    const currentStepIndex = Object.values(AppStep).indexOf(state.currentStep);
    const nextStep = Object.values(AppStep)[currentStepIndex + 1];

    setState({
      ...state,
      currentStep: nextStep
    });

    // Load sample data for demo purposes
    if (nextStep === AppStep.PREVIEW) {
      setState(prevState => ({
        ...prevState,
        previewData: samplePreviewData
      }));
    } else if (nextStep === AppStep.ISSUES) {
      setState(prevState => ({
        ...prevState,
        issues: sampleIssues
      }));
    }
  };

  // Handle previous step
  const handlePreviousStep = () => {
    const currentStepIndex = Object.values(AppStep).indexOf(state.currentStep);
    const previousStep = Object.values(AppStep)[currentStepIndex - 1];

    setState({
      ...state,
      currentStep: previousStep
    });
  };

  // Handle field mapping update
  const handleFieldMappingUpdate = (updatedMappings: FieldMapping[]) => {
    setState({
      ...state,
      fieldMappings: updatedMappings
    });
  };

  // Handle standardization options update
  const handleStandardizationUpdate = (options: StandardizationOptions) => {
    setState({
      ...state,
      standardizationOptions: options
    });
  };

  // Handle process catalog
  const handleProcessCatalog = () => {
    setState({
      ...state,
      isProcessing: true
    });

    // Simulate processing
    setTimeout(() => {
      setState(prevState => ({
        ...prevState,
        isProcessing: false,
        downloadUrl: 'processed-catalog.csv'
      }));
    }, 2000);
  };

  // Render the current step
  const renderCurrentStep = () => {
    switch (state.currentStep) {
      case AppStep.UPLOAD:
        return <UploadStep onFileUpload={handleFileUpload} />;
      case AppStep.FILE_INFO:
        return <FileInfoStep
          fileInfo={state.fileInfo!}
          onNext={handleNextStep}
        />;
      case AppStep.FIELD_MAPPING:
        return <FieldMappingStep
          fieldMappings={state.fieldMappings}
          onUpdate={handleFieldMappingUpdate}
          onNext={handleNextStep}
          onPrevious={handlePreviousStep}
        />;
      case AppStep.STANDARDIZATION:
        return <StandardizationStep
          options={state.standardizationOptions}
          onUpdate={handleStandardizationUpdate}
          onNext={handleNextStep}
          onPrevious={handlePreviousStep}
        />;
      case AppStep.PREVIEW:
        return <PreviewStep
          previewData={state.previewData}
          onNext={handleNextStep}
          onPrevious={handlePreviousStep}
        />;
      case AppStep.ISSUES:
        return <IssuesStep
          issues={state.issues}
          onNext={handleNextStep}
          onPrevious={handlePreviousStep}
        />;
      case AppStep.DOWNLOAD:
        return <DownloadStep
          isProcessing={state.isProcessing}
          downloadUrl={state.downloadUrl}
          onProcess={handleProcessCatalog}
          onPrevious={handlePreviousStep}
        />;
      default:
        return <UploadStep onFileUpload={handleFileUpload} />;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <Box sx={{ mb: 4 }}>
          <Typography variant="h1" component="h1">
            AV Catalog Converter
          </Typography>
        </Box>
        {renderCurrentStep()}
      </Container>
    </ThemeProvider>
  );
}

export default App;
