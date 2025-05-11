# AV Catalog Converter Frontend Guide

This guide provides detailed information about the AV Catalog Converter frontend, including its architecture, components, and development workflow.

## Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Component Architecture](#component-architecture)
- [State Management](#state-management)
- [API Integration](#api-integration)
- [Development Workflow](#development-workflow)
- [Building and Deployment](#building-and-deployment)
- [Customization](#customization)

## Overview

The AV Catalog Converter frontend is a React-based web application that provides a user-friendly interface for uploading, mapping, and processing catalog files. The frontend communicates with the backend API to perform file processing operations.

![Frontend Overview](../images/frontend_overview.png)

## Technology Stack

The frontend is built with the following technologies:

- **React**: JavaScript library for building user interfaces
- **Material-UI**: React component library implementing Google's Material Design
- **Axios**: Promise-based HTTP client for making API requests
- **React Router**: Routing library for React applications
- **React Dropzone**: File upload component
- **React Table**: Table component for displaying data
- **Chart.js**: Charting library for data visualization
- **Jest**: Testing framework

## Project Structure

The frontend project follows a standard React application structure:

```
web/frontend/
├── public/                 # Static files
│   ├── index.html          # HTML template
│   ├── favicon.ico         # Favicon
│   └── ...
├── src/                    # Source code
│   ├── components/         # React components
│   │   ├── UploadStep/     # File upload component
│   │   ├── FileInfoStep/   # File information component
│   │   ├── FieldMappingStep/ # Field mapping component
│   │   ├── StandardizationStep/ # Standardization options component
│   │   ├── PreviewStep/    # Preview component
│   │   ├── IssuesStep/     # Issues display component
│   │   ├── DownloadStep/   # Download component
│   │   └── common/         # Common components
│   ├── services/           # Service modules
│   │   ├── api.js          # API service
│   │   └── utils.js        # Utility functions
│   ├── hooks/              # Custom React hooks
│   ├── context/            # React context providers
│   ├── theme.js            # Theme configuration
│   ├── App.js              # Main application component
│   └── index.js            # Application entry point
├── package.json            # NPM package configuration
└── README.md               # Frontend documentation
```

## Component Architecture

The frontend follows a component-based architecture, with each major feature encapsulated in its own component. The main components are:

### UploadStep

Handles file uploading using React Dropzone. Supports drag-and-drop and file selection.

```jsx
// Example usage
<UploadStep onFileUpload={handleFileUpload} />
```

### FileInfoStep

Displays information about the uploaded file, including file name, size, and detected format.

```jsx
// Example usage
<FileInfoStep fileInfo={fileInfo} />
```

### FieldMappingStep

Allows users to review and adjust the automatically detected field mappings.

```jsx
// Example usage
<FieldMappingStep 
  mappings={mappings} 
  onMappingsChange={handleMappingsChange} 
/>
```

### StandardizationStep

Provides options for standardizing the data, such as currency selection and category mapping.

```jsx
// Example usage
<StandardizationStep 
  options={options} 
  onOptionsChange={handleOptionsChange} 
/>
```

### PreviewStep

Shows a preview of the processed data before final conversion.

```jsx
// Example usage
<PreviewStep previewData={previewData} />
```

### IssuesStep

Displays any issues or warnings detected during processing.

```jsx
// Example usage
<IssuesStep issues={issues} />
```

### DownloadStep

Provides a download link for the processed file.

```jsx
// Example usage
<DownloadStep downloadUrl={downloadUrl} />
```

## State Management

The application uses React's built-in state management capabilities, with state lifted to the App component and passed down to child components via props.

```jsx
// Example state management in App.js
function App() {
  const [state, setState] = useState({
    currentStep: 'upload',
    file: null,
    fileInfo: null,
    mappings: [],
    options: defaultOptions,
    previewData: [],
    issues: [],
    downloadUrl: null,
    error: null
  });

  // State update functions
  const handleFileUpload = (file) => {
    setState(prevState => ({
      ...prevState,
      file,
      currentStep: 'fileInfo'
    }));
  };

  // Render components based on current step
  const renderCurrentStep = () => {
    switch (state.currentStep) {
      case 'upload':
        return <UploadStep onFileUpload={handleFileUpload} />;
      case 'fileInfo':
        return <FileInfoStep fileInfo={state.fileInfo} />;
      // Other steps...
    }
  };

  return (
    <div className="App">
      {renderCurrentStep()}
    </div>
  );
}
```

## API Integration

The frontend communicates with the backend API using the Axios HTTP client. API calls are encapsulated in the `api.js` service module.

```javascript
// Example API service
import axios from 'axios';

const API_PREFIX = '/api';

const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_URL || '',
  headers: {
    'Content-Type': 'application/json'
  }
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

export const apiService = {
  // Check API health
  checkHealth: async () => {
    try {
      const response = await apiClient.get(endpoints.health);
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  },

  // Upload and analyze file
  analyzeFile: async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await axios.post(endpoints.analyze, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('File analysis failed:', error);
      throw error;
    }
  },

  // Other API methods...
};
```

## Development Workflow

### Setting Up the Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/av-catalog-converter.git
   cd av-catalog-converter
   ```

2. Install frontend dependencies:
   ```bash
   cd web/frontend
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. The frontend will be available at http://localhost:3000

### Development Server Configuration

The development server is configured to proxy API requests to the backend server. This is defined in the `package.json` file:

```json
{
  "proxy": "http://localhost:8080"
}
```

### Testing

Run tests using Jest:

```bash
npm test
```

Run tests with coverage:

```bash
npm test -- --coverage
```

## Building and Deployment

### Building for Production

Build the frontend for production:

```bash
npm run build
```

This creates a `build` directory with optimized production files.

### Deployment

The frontend can be deployed in several ways:

1. **Serving with the Backend**: The backend server can serve the frontend static files.

2. **Standalone Deployment**: Deploy the frontend to a static hosting service like Netlify, Vercel, or GitHub Pages.

3. **Docker Deployment**: The frontend is included in the Docker deployment of the full application.

## Customization

### Theming

The application uses Material-UI's theming system. Customize the theme in `src/theme.js`:

```javascript
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    // Other typography settings...
  },
  // Other theme settings...
});

export default theme;
```

### Adding New Components

To add a new component:

1. Create a new directory in `src/components/`
2. Create the component file(s)
3. Export the component
4. Import and use the component in other parts of the application

For more information on frontend development, refer to the React and Material-UI documentation.
