import React, { useState, useEffect } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import {
  Container,
  Box,
  Typography,
  CssBaseline,
  Snackbar,
  Alert,
  CircularProgress
} from '@mui/material';
import theme from './theme';
import UploadStep from './components/UploadStep';
import FileInfoStep from './components/FileInfoStep';
import FieldMappingStep from './components/FieldMappingStep';
import StandardizationStep from './components/StandardizationStep';
import PreviewStep from './components/PreviewStep';
import IssuesStep from './components/IssuesStep';
import DownloadStep from './components/DownloadStep';
import { apiService } from './services/api';

// Define enums and types
const AppStep = {
  UPLOAD: 'upload',
  FILE_INFO: 'file_info',
  FIELD_MAPPING: 'field_mapping',
  STANDARDIZATION: 'standardization',
  PREVIEW: 'preview',
  ISSUES: 'issues',
  DOWNLOAD: 'download'
};

// Initial state
const initialState = {
  currentStep: AppStep.UPLOAD,
  file: null,
  fileInfo: null,
  jobId: null,
  fieldMappings: [],
  standardizationOptions: {
    currency: 'GBP',
    categoryMapping: 'Standard AV Categories',
    removeTax: true,
    includeCurrencySymbol: true
  },
  previewData: [],
  issues: [],
  isProcessing: false,
  downloadUrl: null,
  error: null,
  apiStatus: 'checking' // 'checking', 'ok', 'error'
};

// Sample data for demonstration
const sampleFieldMappings = [
  { sourceColumn: 'Item Name', mapsTo: 'Product Name', confidence: 98, sample: 'HD Camera 4K' },
  { sourceColumn: 'Item #', mapsTo: 'SKU', confidence: 95, sample: 'CAM-4K-01' },
  { sourceColumn: 'Cost (€)', mapsTo: 'Price', confidence: 90, sample: '€299.99' },
  { sourceColumn: 'Category', mapsTo: 'Product Category', confidence: 85, sample: 'Cameras' },
  { sourceColumn: 'Brand', mapsTo: 'Manufacturer', confidence: 92, sample: 'TechVision' },
  { sourceColumn: 'Stock', mapsTo: 'Do not include', confidence: 0, sample: 'In Stock' }
];

const samplePreviewData = [
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

const IssueType = {
  ERROR: 'error',
  WARNING: 'warning',
  INFO: 'info'
};

const sampleIssues = [
  { type: IssueType.WARNING, message: '3 products have missing specifications' },
  { type: IssueType.ERROR, message: '1 duplicate SKU detected' },
  { type: IssueType.INFO, message: '5 prices were converted from EUR to USD' }
];

function App() {
  const [state, setState] = useState(initialState);

  // Check API health on component mount
  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        const healthStatus = await apiService.checkHealth();

        setState(prevState => ({
          ...prevState,
          apiStatus: healthStatus.status === 'ok' ? 'ok' : 'error',
          error: healthStatus.status === 'ok' ? null : 'API is not responding. Please try again later.'
        }));
      } catch (error) {
        setState(prevState => ({
          ...prevState,
          apiStatus: 'error',
          error: 'Failed to connect to the API. Please try again later.'
        }));
      }
    };

    checkApiHealth();
  }, []);

  // Handle file upload
  const handleFileUpload = async (file) => {
    setState({
      ...state,
      file,
      isProcessing: true,
      error: null
    });

    try {
      // Analyze file using real API
      const { fileInfo, jobId } = await apiService.analyzeFile(file);

      setState(prevState => ({
        ...prevState,
        fileInfo,
        jobId,
        isProcessing: false,
        currentStep: AppStep.FILE_INFO
      }));

      // Get field mappings
      const fieldMappings = await apiService.getFieldMappings(jobId);

      setState(prevState => ({
        ...prevState,
        fieldMappings
      }));
    } catch (error) {
      console.error('Error analyzing file:', error);
      setState(prevState => ({
        ...prevState,
        isProcessing: false,
        error: 'Failed to analyze file. Please try again.'
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
      error: null
    });

    // Get preview data when moving to preview step
    if (nextStep === AppStep.PREVIEW && state.jobId) {
      setState(prevState => ({
        ...prevState,
        isProcessing: true
      }));

      try {
        const { previewData, issues } = await apiService.getPreviewData(
          state.jobId,
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
        setState(prevState => ({
          ...prevState,
          isProcessing: false,
          error: 'Failed to generate preview. Please try again.'
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
  const handleFieldMappingUpdate = async (updatedMappings) => {
    setState({
      ...state,
      fieldMappings: updatedMappings,
      isProcessing: true,
      error: null
    });

    try {
      // Update field mappings on the server
      await apiService.updateFieldMappings(state.jobId, updatedMappings);

      setState(prevState => ({
        ...prevState,
        isProcessing: false
      }));
    } catch (error) {
      console.error('Error updating field mappings:', error);
      setState(prevState => ({
        ...prevState,
        isProcessing: false,
        error: 'Failed to update field mappings. Please try again.'
      }));
    }
  };

  // Handle standardization options update
  const handleStandardizationUpdate = (options) => {
    setState({
      ...state,
      standardizationOptions: options
    });
  };

  // Handle process catalog
  const handleProcessCatalog = async () => {
    setState({
      ...state,
      isProcessing: true,
      error: null
    });

    try {
      // Process catalog using real API
      const downloadUrl = await apiService.processCatalog(
        state.jobId,
        state.standardizationOptions
      );

      setState(prevState => ({
        ...prevState,
        isProcessing: false,
        downloadUrl
      }));
    } catch (error) {
      console.error('Error processing catalog:', error);
      setState(prevState => ({
        ...prevState,
        isProcessing: false,
        error: 'Failed to process catalog. Please try again.'
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

        {/* API Status Checking */}
        {state.apiStatus === 'checking' && (
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', my: 8 }}>
            <CircularProgress size={60} sx={{ mb: 2 }} />
            <Typography variant="h6">Connecting to API...</Typography>
          </Box>
        )}

        {/* Processing Indicator */}
        {state.isProcessing && (
          <Box sx={{ display: 'flex', justifyContent: 'center', mb: 4 }}>
            <Typography>Processing... Please wait.</Typography>
          </Box>
        )}

        {/* Main Content - Only show when API is OK or we're in development mode */}
        {(state.apiStatus === 'ok' || process.env.NODE_ENV === 'development') && renderCurrentStep()}

        {/* Error Snackbar */}
        <Snackbar
          open={!!state.error}
          autoHideDuration={6000}
          onClose={() => setState(prev => ({ ...prev, error: null }))}
        >
          <Alert
            onClose={() => setState(prev => ({ ...prev, error: null }))}
            severity="error"
            sx={{ width: '100%' }}
          >
            {state.error}
          </Alert>
        </Snackbar>
      </Container>
    </ThemeProvider>
  );
}

export default App;
