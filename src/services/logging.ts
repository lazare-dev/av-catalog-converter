/**
 * Frontend Logging Service
 * 
 * Provides comprehensive logging capabilities for the frontend application.
 * Logs are sent to both the console and the backend API.
 */
import axios from 'axios';

// Base API URL - change this to your backend URL
const API_BASE_URL = 'http://localhost:8080/api';

// Log levels
export enum LogLevel {
  DEBUG = 'DEBUG',
  INFO = 'INFO',
  WARNING = 'WARNING',
  ERROR = 'ERROR',
  CRITICAL = 'CRITICAL'
}

// Log entry interface
export interface LogEntry {
  timestamp: string;
  level: LogLevel;
  component: string;
  message: string;
  context?: Record<string, any>;
  sessionId?: string;
  userId?: string;
}

// Configuration for the logging service
export interface LoggingConfig {
  minLevel: LogLevel;
  sendToBackend: boolean;
  batchSize: number;
  batchIntervalMs: number;
  includeContext: boolean;
  consoleOutput: boolean;
}

// Default configuration
const DEFAULT_CONFIG: LoggingConfig = {
  minLevel: LogLevel.INFO,
  sendToBackend: true,
  batchSize: 10,
  batchIntervalMs: 5000, // 5 seconds
  includeContext: true,
  consoleOutput: true
};

class LoggingService {
  private config: LoggingConfig;
  private logQueue: LogEntry[] = [];
  private batchTimer: NodeJS.Timeout | null = null;
  private sessionId: string;
  private userId: string | null = null;

  constructor(config: Partial<LoggingConfig> = {}) {
    // Merge provided config with defaults
    this.config = { ...DEFAULT_CONFIG, ...config };
    
    // Generate a session ID
    this.sessionId = this.generateSessionId();
    
    // Start batch timer
    this.startBatchTimer();
    
    // Log initialization
    this.info('LoggingService', 'Logging service initialized', { 
      config: this.config,
      sessionId: this.sessionId
    });
  }

  /**
   * Log a debug message
   */
  public debug(component: string, message: string, context?: Record<string, any>): void {
    this.log(LogLevel.DEBUG, component, message, context);
  }

  /**
   * Log an info message
   */
  public info(component: string, message: string, context?: Record<string, any>): void {
    this.log(LogLevel.INFO, component, message, context);
  }

  /**
   * Log a warning message
   */
  public warning(component: string, message: string, context?: Record<string, any>): void {
    this.log(LogLevel.WARNING, component, message, context);
  }

  /**
   * Log an error message
   */
  public error(component: string, message: string, context?: Record<string, any>): void {
    this.log(LogLevel.ERROR, component, message, context);
  }

  /**
   * Log a critical message
   */
  public critical(component: string, message: string, context?: Record<string, any>): void {
    this.log(LogLevel.CRITICAL, component, message, context);
  }

  /**
   * Set the user ID for the current session
   */
  public setUserId(userId: string): void {
    this.userId = userId;
    this.info('LoggingService', 'User ID set', { userId });
  }

  /**
   * Update the logging configuration
   */
  public updateConfig(config: Partial<LoggingConfig>): void {
    this.config = { ...this.config, ...config };
    this.info('LoggingService', 'Configuration updated', { config: this.config });
    
    // Restart batch timer with new interval if it changed
    if (config.batchIntervalMs !== undefined) {
      this.startBatchTimer();
    }
  }

  /**
   * Flush all pending logs immediately
   */
  public flush(): Promise<void> {
    return this.sendLogs();
  }

  /**
   * Internal method to log a message
   */
  private log(level: LogLevel, component: string, message: string, context?: Record<string, any>): void {
    // Skip if below minimum level
    if (this.getLevelValue(level) < this.getLevelValue(this.config.minLevel)) {
      return;
    }

    const timestamp = new Date().toISOString();
    
    // Create log entry
    const logEntry: LogEntry = {
      timestamp,
      level,
      component,
      message,
      sessionId: this.sessionId,
      userId: this.userId || undefined
    };

    // Add context if enabled
    if (this.config.includeContext && context) {
      logEntry.context = this.sanitizeContext(context);
    }

    // Output to console if enabled
    if (this.config.consoleOutput) {
      this.logToConsole(logEntry);
    }

    // Add to queue for backend sending
    if (this.config.sendToBackend) {
      this.logQueue.push(logEntry);
      
      // Send immediately if batch size reached
      if (this.logQueue.length >= this.config.batchSize) {
        this.sendLogs();
      }
    }
  }

  /**
   * Output a log entry to the console
   */
  private logToConsole(entry: LogEntry): void {
    const timestamp = entry.timestamp.split('T')[1].replace('Z', '');
    const prefix = `[${timestamp}] [${entry.level}] [${entry.component}]`;
    
    switch (entry.level) {
      case LogLevel.DEBUG:
        console.debug(prefix, entry.message, entry.context || '');
        break;
      case LogLevel.INFO:
        console.info(prefix, entry.message, entry.context || '');
        break;
      case LogLevel.WARNING:
        console.warn(prefix, entry.message, entry.context || '');
        break;
      case LogLevel.ERROR:
      case LogLevel.CRITICAL:
        console.error(prefix, entry.message, entry.context || '');
        break;
    }
  }

  /**
   * Send logs to the backend
   */
  private async sendLogs(): Promise<void> {
    if (this.logQueue.length === 0) {
      return;
    }

    // Get logs to send and clear queue
    const logsToSend = [...this.logQueue];
    this.logQueue = [];

    try {
      await axios.post(`${API_BASE_URL}/logs`, {
        logs: logsToSend,
        sessionId: this.sessionId,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent
      });
    } catch (error) {
      // Log to console only to avoid infinite loop
      console.error('Failed to send logs to backend:', error);
      
      // Put logs back in queue for retry
      this.logQueue = [...logsToSend, ...this.logQueue];
    }
  }

  /**
   * Start or restart the batch timer
   */
  private startBatchTimer(): void {
    if (this.batchTimer) {
      clearInterval(this.batchTimer);
    }
    
    this.batchTimer = setInterval(() => {
      if (this.logQueue.length > 0) {
        this.sendLogs();
      }
    }, this.config.batchIntervalMs);
  }

  /**
   * Generate a unique session ID
   */
  private generateSessionId(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = Math.random() * 16 | 0;
      const v = c === 'x' ? r : (r & 0x3 | 0x8);
      return v.toString(16);
    });
  }

  /**
   * Get numeric value for log level for comparison
   */
  private getLevelValue(level: LogLevel): number {
    switch (level) {
      case LogLevel.DEBUG: return 0;
      case LogLevel.INFO: return 1;
      case LogLevel.WARNING: return 2;
      case LogLevel.ERROR: return 3;
      case LogLevel.CRITICAL: return 4;
      default: return 0;
    }
  }

  /**
   * Sanitize context object to ensure it's serializable
   */
  private sanitizeContext(context: Record<string, any>): Record<string, any> {
    const sanitized: Record<string, any> = {};
    
    for (const [key, value] of Object.entries(context)) {
      if (value === undefined) {
        sanitized[key] = 'undefined';
      } else if (value === null) {
        sanitized[key] = null;
      } else if (typeof value === 'function') {
        sanitized[key] = '[Function]';
      } else if (typeof value === 'object') {
        if (value instanceof Error) {
          sanitized[key] = {
            name: value.name,
            message: value.message,
            stack: value.stack
          };
        } else if (Array.isArray(value)) {
          sanitized[key] = value.map(item => 
            typeof item === 'object' ? '[Object]' : item
          );
        } else {
          try {
            // Test if serializable
            JSON.stringify(value);
            sanitized[key] = value;
          } catch (e) {
            sanitized[key] = '[Unserializable Object]';
          }
        }
      } else {
        sanitized[key] = value;
      }
    }
    
    return sanitized;
  }
}

// Create and export singleton instance
export const logger = new LoggingService();

// Export class for testing or custom instances
export default LoggingService;
