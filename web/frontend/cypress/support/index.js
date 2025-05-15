// ***********************************************************
// This support file is processed and loaded automatically before your test files.
//
// You can read more here:
// https://on.cypress.io/configuration
// ***********************************************************

// Import commands.js using ES2015 syntax:
import './commands'

// Alternatively you can use CommonJS syntax:
// require('./commands')

// Log Cypress events to console for debugging
Cypress.on('log:added', (log) => {
  if (log.displayName === 'xhr' || log.displayName === 'fetch') {
    console.log(`${log.displayName} ${log.url || log.message}`);
  }
});

// Log uncaught exceptions
Cypress.on('uncaught:exception', (err, runnable) => {
  console.error('Uncaught exception:', err.message);
  return false; // returning false prevents Cypress from failing the test
});
