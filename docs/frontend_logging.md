# Frontend Logging System Documentation

## Overview

The AV Catalog Converter frontend includes a comprehensive logging system that captures user interactions, application state changes, API calls, and errors. This logging system helps with debugging, performance monitoring, and understanding user behavior.

## Architecture

The logging system consists of the following components:

1. **Frontend Logging Service**: A TypeScript service that captures logs on the client side
2. **Backend Logging API**: A Flask endpoint that receives logs from the frontend
3. **Integrated Logging**: Components throughout the application use the logging service

## Log Levels

The logging system supports the following log levels, in order of increasing severity:

- **DEBUG**: Detailed information for debugging purposes
- **INFO**: General information about application operation
- **WARNING**: Potential issues that don't prevent the application from working
- **ERROR**: Errors that prevent specific operations from working
- **CRITICAL**: Critical errors that may cause the application to fail

## Using the Logging Service

### Basic Usage

```typescript
import { logger } from '../services/logging';

// Log at different levels
logger.debug('ComponentName', 'Detailed debug message');
logger.info('ComponentName', 'General information');
logger.warning('ComponentName', 'Warning message');
logger.error('ComponentName', 'Error message');
logger.critical('ComponentName', 'Critical error message');
```

### Adding Context

You can add additional context to log messages:

```typescript
logger.info('ComponentName', 'User selected a file', {
  fileName: file.name,
  fileSize: file.size,
  fileType: file.type
});
```

### Logging in React Components

For React components, it's recommended to log component lifecycle events:

```typescript
import React, { useEffect } from 'react';
import { logger } from '../services/logging';

const MyComponent = () => {
  useEffect(() => {
    // Log component mount
    logger.info('MyComponent', 'Component mounted');
    
    // Log component unmount
    return () => {
      logger.info('MyComponent', 'Component unmounted');
    };
  }, []);
  
  // Log user interactions
  const handleButtonClick = () => {
    logger.info('MyComponent', 'Button clicked');
    // Component logic...
  };
  
  return (
    <button onClick={handleButtonClick}>Click Me</button>
  );
};
```

## Configuration

The logging service can be configured with the following options:

- **minLevel**: Minimum log level to capture (default: INFO)
- **sendToBackend**: Whether to send logs to the backend (default: true)
- **batchSize**: Number of logs to batch before sending (default: 10)
- **batchIntervalMs**: Interval in milliseconds to send batched logs (default: 5000)
- **includeContext**: Whether to include context data (default: true)
- **consoleOutput**: Whether to output logs to the console (default: true)

Example:

```typescript
import { logger, LogLevel } from '../services/logging';

// Update configuration
logger.updateConfig({
  minLevel: LogLevel.DEBUG,
  batchIntervalMs: 10000
});
```

## Backend Integration

The frontend logging service automatically sends logs to the backend API endpoint `/api/logs`. The backend processes these logs and integrates them with the server-side logging system.

Each log entry sent to the backend includes:

- Timestamp
- Log level
- Component name
- Message
- Context data (if enabled)
- Session ID
- User ID (if available)

## Log Format

Frontend logs are formatted as JSON objects with the following structure:

```json
{
  "timestamp": "2025-05-12T15:30:45.123Z",
  "level": "INFO",
  "component": "ComponentName",
  "message": "Log message",
  "context": {
    "key1": "value1",
    "key2": "value2"
  },
  "sessionId": "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx",
  "userId": "user123"
}
```

## Performance Considerations

The logging system is designed to have minimal impact on application performance:

- Logs are batched to reduce API calls
- Context objects are sanitized to ensure they're serializable
- Log levels allow filtering out less important logs
- Logs below the minimum level are discarded immediately

## Troubleshooting

If logs are not appearing in the backend:

1. Check that the frontend logging service is properly initialized
2. Verify that the backend API endpoint is accessible
3. Check for console errors related to log transmission
4. Ensure the log level is set appropriately

## Best Practices

1. **Be Consistent**: Use consistent component names and message formats
2. **Log Meaningful Events**: Focus on logging important user actions and state changes
3. **Include Context**: Add relevant context data to make logs more useful
4. **Use Appropriate Levels**: Use the correct log level for each message
5. **Don't Log Sensitive Data**: Avoid logging passwords, tokens, or personal information
6. **Log Start and End**: Log both the beginning and completion of important operations

## Example Logging Patterns

### API Calls

```typescript
// Before making API call
logger.info('ServiceName', 'Making API request', { endpoint: '/api/data', params });

try {
  const response = await apiClient.get('/api/data', { params });
  logger.info('ServiceName', 'API request successful', { 
    endpoint: '/api/data',
    responseStatus: response.status,
    itemCount: response.data.length
  });
  return response.data;
} catch (error) {
  logger.error('ServiceName', 'API request failed', { 
    endpoint: '/api/data',
    error: error.message,
    status: error.response?.status
  });
  throw error;
}
```

### User Interactions

```typescript
const handleSubmit = (formData) => {
  logger.info('FormComponent', 'Form submitted', {
    formName: 'userProfile',
    fieldCount: Object.keys(formData).length
  });
  
  // Process form...
};
```

### Error Handling

```typescript
try {
  // Operation that might fail
} catch (error) {
  logger.error('ComponentName', 'Operation failed', {
    operation: 'processingFile',
    error: error.message,
    stack: error.stack
  });
  
  // Show error to user...
}
```
