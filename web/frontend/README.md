# AV Catalog Converter Frontend

This is the React frontend for the AV Catalog Converter application. It provides a clean, step-by-step interface for converting and standardizing audio-visual equipment catalogs.

## Features

- Clean, uncluttered UI design
- Step-by-step workflow showing one section at a time
- File upload with drag-and-drop support
- Automatic field mapping
- Standardization options
- Preview and validation
- Issue detection and reporting

## Getting Started

### Prerequisites

- Node.js 14.x or higher
- npm 6.x or higher

### Installation

1. Navigate to the frontend directory:
   ```
   cd web/frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:3000
   ```

## Project Structure

```
frontend/
├── public/                # Static files
├── src/                   # Source code
│   ├── components/        # React components
│   │   ├── UploadStep.tsx        # File upload step
│   │   ├── FileInfoStep.tsx      # File information step
│   │   ├── FieldMappingStep.tsx  # Field mapping step
│   │   ├── StandardizationStep.tsx # Standardization options step
│   │   ├── PreviewStep.tsx       # Data preview step
│   │   ├── IssuesStep.tsx        # Issues and warnings step
│   │   └── DownloadStep.tsx      # Process and download step
│   ├── App.tsx            # Main application component
│   ├── index.tsx          # Application entry point
│   ├── theme.ts           # Material-UI theme configuration
│   └── types.ts           # TypeScript type definitions
└── package.json           # Project dependencies and scripts
```

## User Flow

1. **Upload**: User uploads a catalog file (CSV, Excel, PDF, etc.)
2. **File Info**: System displays basic information about the file
3. **Field Mapping**: System suggests mappings from source columns to standard fields
4. **Standardization**: User selects options for standardizing data
5. **Preview**: User previews the processed data
6. **Issues**: System displays any issues or warnings
7. **Download**: User processes the catalog and downloads the result

## Available Scripts

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

### `npm test`

Launches the test runner in the interactive watch mode.

### `npm run build`

Builds the app for production to the `build` folder.

## Connecting to Backend

The frontend is designed to connect to the AV Catalog Converter backend API. In a production environment, you would:

1. Replace the sample data with actual API calls
2. Implement file upload to the backend
3. Process the catalog using the backend services

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).
