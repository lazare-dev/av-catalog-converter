import React, { useEffect } from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { Container, CssBaseline } from '@mui/material';
import theme from './theme';
import SimpleCatalogConverter from './components/SimpleCatalogConverter';
import { logger } from './services/logging';

function App() {
  // Log application startup
  useEffect(() => {
    logger.info('App', 'Application initialized', {
      userAgent: navigator.userAgent,
      screenSize: `${window.innerWidth}x${window.innerHeight}`,
      timestamp: new Date().toISOString()
    });

    // Log when app is closed/unloaded
    const handleUnload = () => {
      logger.info('App', 'Application closing');
    };

    window.addEventListener('beforeunload', handleUnload);

    return () => {
      window.removeEventListener('beforeunload', handleUnload);
    };
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{ py: 4 }}>
        <SimpleCatalogConverter />
      </Container>
    </ThemeProvider>
  );
}

export default App;
