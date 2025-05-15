import axios from 'axios';
import LoggingService, { LogLevel, logger } from './logging';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('LoggingService', () => {
  let originalConsole: any;
  let consoleOutput: any = {};
  
  // Setup console mock
  beforeEach(() => {
    // Save original console methods
    originalConsole = {
      debug: console.debug,
      info: console.info,
      warn: console.warn,
      error: console.error
    };
    
    // Mock console methods
    console.debug = jest.fn((...args) => { consoleOutput.debug = args; });
    console.info = jest.fn((...args) => { consoleOutput.info = args; });
    console.warn = jest.fn((...args) => { consoleOutput.warn = args; });
    console.error = jest.fn((...args) => { consoleOutput.error = args; });
    
    // Clear console output
    consoleOutput = {};
    
    // Mock Date.now for consistent timestamps
    jest.spyOn(Date.prototype, 'toISOString').mockReturnValue('2025-05-12T12:00:00.000Z');
    
    // Clear all mocks
    jest.clearAllMocks();
  });
  
  // Restore console
  afterEach(() => {
    console.debug = originalConsole.debug;
    console.info = originalConsole.info;
    console.warn = originalConsole.warn;
    console.error = originalConsole.error;
    
    jest.restoreAllMocks();
  });
  
  test('creates a new instance with default config', () => {
    const loggingService = new LoggingService();
    expect(loggingService).toBeDefined();
  });
  
  test('logs messages at different levels', () => {
    const loggingService = new LoggingService({
      minLevel: LogLevel.DEBUG,
      sendToBackend: false
    });
    
    loggingService.debug('TestComponent', 'Debug message');
    expect(console.debug).toHaveBeenCalled();
    expect(consoleOutput.debug[0]).toContain('[DEBUG]');
    expect(consoleOutput.debug[0]).toContain('[TestComponent]');
    expect(consoleOutput.debug[1]).toBe('Debug message');
    
    loggingService.info('TestComponent', 'Info message');
    expect(console.info).toHaveBeenCalled();
    expect(consoleOutput.info[0]).toContain('[INFO]');
    
    loggingService.warning('TestComponent', 'Warning message');
    expect(console.warn).toHaveBeenCalled();
    expect(consoleOutput.warn[0]).toContain('[WARNING]');
    
    loggingService.error('TestComponent', 'Error message');
    expect(console.error).toHaveBeenCalled();
    expect(consoleOutput.error[0]).toContain('[ERROR]');
    
    loggingService.critical('TestComponent', 'Critical message');
    expect(console.error).toHaveBeenCalled();
    expect(consoleOutput.error[0]).toContain('[CRITICAL]');
  });
  
  test('respects minimum log level', () => {
    const loggingService = new LoggingService({
      minLevel: LogLevel.WARNING,
      sendToBackend: false
    });
    
    loggingService.debug('TestComponent', 'Debug message');
    expect(console.debug).not.toHaveBeenCalled();
    
    loggingService.info('TestComponent', 'Info message');
    expect(console.info).not.toHaveBeenCalled();
    
    loggingService.warning('TestComponent', 'Warning message');
    expect(console.warn).toHaveBeenCalled();
    
    loggingService.error('TestComponent', 'Error message');
    expect(console.error).toHaveBeenCalled();
  });
  
  test('includes context in log messages', () => {
    const loggingService = new LoggingService({
      minLevel: LogLevel.INFO,
      sendToBackend: false,
      includeContext: true
    });
    
    const context = { userId: '123', action: 'test' };
    loggingService.info('TestComponent', 'Info with context', context);
    
    expect(console.info).toHaveBeenCalled();
    expect(consoleOutput.info[2]).toEqual(context);
  });
  
  test('sends logs to backend', async () => {
    // Mock successful axios post
    mockedAxios.post.mockResolvedValue({ data: { success: true } });
    
    const loggingService = new LoggingService({
      minLevel: LogLevel.INFO,
      sendToBackend: true,
      batchSize: 2, // Small batch size for testing
      batchIntervalMs: 1000
    });
    
    // Log two messages to trigger batch send
    loggingService.info('TestComponent', 'First message');
    loggingService.info('TestComponent', 'Second message');
    
    // Wait for axios to be called
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Check that axios.post was called with the correct data
    expect(mockedAxios.post).toHaveBeenCalledTimes(1);
    expect(mockedAxios.post).toHaveBeenCalledWith(
      expect.stringContaining('/logs'),
      expect.objectContaining({
        logs: expect.arrayContaining([
          expect.objectContaining({
            level: LogLevel.INFO,
            component: 'TestComponent',
            message: 'First message'
          }),
          expect.objectContaining({
            level: LogLevel.INFO,
            component: 'TestComponent',
            message: 'Second message'
          })
        ])
      })
    );
  });
  
  test('handles backend errors gracefully', async () => {
    // Mock failed axios post
    mockedAxios.post.mockRejectedValue(new Error('Network error'));
    
    const loggingService = new LoggingService({
      minLevel: LogLevel.INFO,
      sendToBackend: true,
      batchSize: 1,
      batchIntervalMs: 1000
    });
    
    // Spy on console.error to check error handling
    const consoleErrorSpy = jest.spyOn(console, 'error');
    
    // Log a message to trigger send
    loggingService.info('TestComponent', 'Test message');
    
    // Wait for axios to be called
    await new Promise(resolve => setTimeout(resolve, 100));
    
    // Check that error was logged to console
    expect(consoleErrorSpy).toHaveBeenCalledWith(
      expect.stringContaining('Failed to send logs to backend:'),
      expect.any(Error)
    );
  });
  
  test('sanitizes context objects', () => {
    const loggingService = new LoggingService({
      minLevel: LogLevel.INFO,
      sendToBackend: false
    });
    
    // Create a circular reference
    const circular: any = { name: 'circular' };
    circular.self = circular;
    
    // Create a context with various types
    const context = {
      string: 'test',
      number: 123,
      boolean: true,
      null: null,
      undefined: undefined,
      function: () => 'test',
      circular,
      largeArray: Array(20).fill('item'),
      error: new Error('Test error')
    };
    
    loggingService.info('TestComponent', 'Complex context', context);
    
    // Check that console.info was called
    expect(console.info).toHaveBeenCalled();
    
    // Check that context was sanitized
    const sanitizedContext = consoleOutput.info[2];
    expect(sanitizedContext.string).toBe('test');
    expect(sanitizedContext.number).toBe(123);
    expect(sanitizedContext.boolean).toBe(true);
    expect(sanitizedContext.null).toBe(null);
    expect(sanitizedContext.function).toBe('[Function]');
    expect(sanitizedContext.largeArray).toEqual(expect.any(Array));
    expect(sanitizedContext.error).toEqual(expect.objectContaining({
      name: 'Error',
      message: 'Test error'
    }));
  });
  
  test('updates configuration', () => {
    const loggingService = new LoggingService({
      minLevel: LogLevel.INFO
    });
    
    // Log a debug message (should be filtered out)
    loggingService.debug('TestComponent', 'Debug message');
    expect(console.debug).not.toHaveBeenCalled();
    
    // Update config to include debug messages
    loggingService.updateConfig({
      minLevel: LogLevel.DEBUG
    });
    
    // Log another debug message (should be logged)
    loggingService.debug('TestComponent', 'Debug message after config update');
    expect(console.debug).toHaveBeenCalled();
  });
  
  test('singleton logger instance works correctly', () => {
    // The exported logger should be a singleton instance
    expect(logger).toBeInstanceOf(LoggingService);
    
    // It should log messages
    logger.info('TestComponent', 'Test singleton');
    expect(console.info).toHaveBeenCalled();
  });
});
