import React, { useState } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { Container, Box, Typography, CssBaseline, Snackbar, Alert } from '@mui/material';
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
import { apiService } from './services/api';

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
  const handleFileUpload = async (file: File) => {
    try {
      setState({
        ...state,
        file,
        currentStep: AppStep.FILE_INFO,
        isProcessing: true
      });

      // Call the API to analyze the file
      const { fileInfo, structure } = await apiService.analyzeFile(file);

      // Update state with file info
      setState(prevState => ({
        ...prevState,
        fileInfo,
        isProcessing: false
      }));

      // Get field mappings from the API
      if (structure && structure.columns) {
        const mappings = await apiService.getFieldMappings(
          structure.columns,
          structure.sample_data || []
        );

        setState(prevState => ({
          ...prevState,
          fieldMappings: mappings
        }));
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Failed to upload file. Please try again.');
      setState(prevState => ({
        ...prevState,
        isProcessing: false
      }));
    }
  };

  // Handle next step
  const handleNextStep = async () => {
    const currentStepIndex = Object.values(AppStep).indexOf(state.currentStep);
    const nextStep = Object.values(AppStep)[currentStepIndex + 1];

    setState({
      ...state,
      currentStep: nextStep,
      isProcessing: nextStep === AppStep.PREVIEW
    });

    // Get preview data from API when moving to preview step
    if (nextStep === AppStep.PREVIEW && state.file) {
      try {
        const { previewData, issues } = await apiService.getPreviewData(
          state.file,
          state.fieldMappings,
          state.standardizationOptions
        );

        setState(prevState => ({
          ...prevState,
          previewData,
          issues,
          isProcessing: false
        }));
      } catch (error) {
        console.error('Error generating preview:', error);
        alert('Failed to generate preview. Please try again.');
        setState(prevState => ({
          ...prevState,
          isProcessing: false
        }));
      }
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
  const handleProcessCatalog = async () => {
    if (!state.file) {
      alert('No file to process. Please upload a file first.');
      return;
    }

    setState({
      ...state,
      isProcessing: true
    });

    try {
      // Process the file with the API
      const downloadUrl = await apiService.processFile(
        state.file,
        state.fieldMappings,
        state.standardizationOptions
      );

      setState(prevState => ({
        ...prevState,
        isProcessing: false,
        downloadUrl
      }));
    } catch (error) {
      console.error('Error processing catalog:', error);
      alert('Failed to process catalog. Please try again.');
      setState(prevState => ({
        ...prevState,
        isProcessing: false
      }));
    }
  };

  // Render the current step
  const renderCurrentStep = () => {
    switch (state.currentStep) {
      case AppStep.UPLOAD:
        return <UploadStep onFileUpload={handleFileUpload} />;
      case AppStep.FILE_INFO:
        return <FileInfoStep
          fileInfo={state.fileInfo}
          onNext={handleNextStep}
          isLoading={state.isProcessing}
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
          isLoading={state.isProcessing}
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
