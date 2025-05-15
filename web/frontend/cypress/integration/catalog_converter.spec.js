/// <reference types="cypress" />

describe('AV Catalog Converter Integration Tests', () => {
  beforeEach(() => {
    // Visit the app before each test
    cy.visit('http://localhost:3000');
    
    // Wait for app to load
    cy.contains('AV Catalog Converter').should('be.visible');
  });

  it('should display the initial upload screen', () => {
    // Check that the upload button is visible
    cy.contains('Select Catalog File').should('be.visible');
    
    // Check that the stepper shows the correct steps
    cy.contains('Upload File').should('be.visible');
    cy.contains('Validate').should('be.visible');
    cy.contains('Export').should('be.visible');
  });

  it('should allow file upload and validation', () => {
    // Create a test file
    cy.fixture('test_catalog.csv').then(fileContent => {
      // Convert the file content to a blob
      const blob = Cypress.Blob.base64StringToBlob(
        btoa(fileContent),
        'text/csv'
      );
      
      // Create a File object
      const testFile = new File([blob], 'test_catalog.csv', { type: 'text/csv' });
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(testFile);
      
      // Attach the file to the hidden input
      cy.get('input[type="file"]').then(input => {
        input[0].files = dataTransfer.files;
        cy.wrap(input).trigger('change', { force: true });
      });
    });
    
    // Check that the file name is displayed
    cy.contains('test_catalog.csv').should('be.visible');
    
    // Select output format
    cy.get('[aria-labelledby="output-format-label"]').click();
    cy.contains('CSV').click();
    
    // Click validate button
    cy.contains('Validate File').click();
    
    // Wait for validation to complete
    cy.contains('Validation Results', { timeout: 10000 }).should('be.visible');
    
    // Check that file info is displayed
    cy.contains('File Information').should('be.visible');
    cy.contains('Name:').should('be.visible');
    
    // Check that the process button is enabled
    cy.contains('Process & Export').should('not.be.disabled');
  });

  it('should process and export the file', () => {
    // Upload file first
    cy.fixture('test_catalog.csv').then(fileContent => {
      const blob = Cypress.Blob.base64StringToBlob(
        btoa(fileContent),
        'text/csv'
      );
      const testFile = new File([blob], 'test_catalog.csv', { type: 'text/csv' });
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(testFile);
      
      cy.get('input[type="file"]').then(input => {
        input[0].files = dataTransfer.files;
        cy.wrap(input).trigger('change', { force: true });
      });
    });
    
    // Validate file
    cy.contains('Validate File').click();
    cy.contains('Validation Results', { timeout: 10000 }).should('be.visible');
    
    // Process file
    cy.contains('Process & Export').click();
    
    // Wait for processing to complete
    cy.contains('Processing Complete!', { timeout: 15000 }).should('be.visible');
    
    // Check that download button is displayed
    cy.contains('Download Converted File').should('be.visible');
  });

  it('should handle validation errors', () => {
    // Create an empty file
    cy.fixture('empty_catalog.csv').then(fileContent => {
      const blob = Cypress.Blob.base64StringToBlob(
        btoa(fileContent),
        'text/csv'
      );
      const testFile = new File([blob], 'empty_catalog.csv', { type: 'text/csv' });
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(testFile);
      
      cy.get('input[type="file"]').then(input => {
        input[0].files = dataTransfer.files;
        cy.wrap(input).trigger('change', { force: true });
      });
    });
    
    // Validate file
    cy.contains('Validate File').click();
    
    // Wait for validation to complete
    cy.contains('Validation Results', { timeout: 10000 }).should('be.visible');
    
    // Check that error message is displayed
    cy.contains('No products detected in the file').should('be.visible');
    
    // Check that the process button is disabled
    cy.contains('Process & Export').should('be.disabled');
  });

  it('should log frontend actions to backend', () => {
    // Intercept the logs API call
    cy.intercept('POST', '/api/logs').as('logRequest');
    
    // Perform some actions
    cy.contains('Select Catalog File').click();
    
    // Wait for log request
    cy.wait('@logRequest').then(interception => {
      // Check that logs were sent
      expect(interception.request.body).to.have.property('logs');
      expect(interception.request.body.logs).to.be.an('array');
      
      // Check log content
      const logs = interception.request.body.logs;
      expect(logs.some(log => 
        log.component === 'SimpleCatalogConverter' && 
        log.level === 'INFO'
      )).to.be.true;
    });
  });

  it('should handle API errors gracefully', () => {
    // Intercept the analyze API call and force it to fail
    cy.intercept('POST', '/api/analyze', {
      statusCode: 500,
      body: {
        error: 'Server error',
        details: 'Test error'
      }
    }).as('analyzeRequest');
    
    // Upload file
    cy.fixture('test_catalog.csv').then(fileContent => {
      const blob = Cypress.Blob.base64StringToBlob(
        btoa(fileContent),
        'text/csv'
      );
      const testFile = new File([blob], 'test_catalog.csv', { type: 'text/csv' });
      const dataTransfer = new DataTransfer();
      dataTransfer.items.add(testFile);
      
      cy.get('input[type="file"]').then(input => {
        input[0].files = dataTransfer.files;
        cy.wrap(input).trigger('change', { force: true });
      });
    });
    
    // Validate file
    cy.contains('Validate File').click();
    
    // Wait for API call
    cy.wait('@analyzeRequest');
    
    // Check that error message is displayed
    cy.contains('File validation failed').should('be.visible');
  });
});
