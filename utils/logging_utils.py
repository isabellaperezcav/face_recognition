"""
Logging utilities for the AI Detection System.

This module provides logging configuration and utilities for the application,
including:
- Console logging with color support
- File logging with rotation
- Custom formatters and handlers
- Log level management

Usage:
    from utils.logging_utils import setup_logger
    
    # Setup logger for a module
    logger = setup_logger('my_module')
    
    # Log messages
    logger.info('This is an info message')
    logger.warning('This is a warning')
    logger.error('This is an error')
"""

import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import datetime
from typing import Optional, Dict, Any, Union

# ANSI color codes for colored terminal output
COLORS = {
    'RESET': '\033[0m',
    'BLACK': '\033[30m',
    'RED': '\033[31m',
    'GREEN': '\033[32m',
    'YELLOW': '\033[33m',
    'BLUE': '\033[34m',
    'MAGENTA': '\033[35m',
    'CYAN': '\033[36m',
    'WHITE': '\033[37m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m'
}

# Log level to color mapping
LEVEL_COLORS = {
    'DEBUG': COLORS['BLUE'],
    'INFO': COLORS['GREEN'],
    'WARNING': COLORS['YELLOW'],
    'ERROR': COLORS['RED'],
    'CRITICAL': COLORS['BOLD'] + COLORS['RED']
}

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter to add colors to log messages in the console.
    """
    
    def format(self, record):
        # Save original values
        orig_levelname = record.levelname
        orig_msg = record.msg
        
        # Add color to levelname if it exists in our mapping
        if record.levelname in LEVEL_COLORS:
            record.levelname = f"{LEVEL_COLORS[record.levelname]}{record.levelname}{COLORS['RESET']}"
        
        # Format the message
        result = super().format(record)
        
        # Restore original values
        record.levelname = orig_levelname
        record.msg = orig_msg
        
        return result

def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    log_dir: str = 'logs',
    app_name: str = 'ai_detection',
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console: bool = True,
    colored: bool = True
) -> None:
    """
    Setup global logging configuration.
    
    Args:
        log_level: The log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: The log file name (optional)
        log_dir: The directory for log files
        app_name: The application name (used as prefix for log files)
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
        console: Whether to log to console
        colored: Whether to use colored output in console
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    file_format = '%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s'
    
    if colored and console:
        console_formatter = ColoredFormatter(console_format, datefmt='%Y-%m-%d %H:%M:%S')
    else:
        console_formatter = logging.Formatter(console_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    file_formatter = logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    # Setup console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # Setup file handler if requested
    if log_file is not None:
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Generate log file path
        log_path = os.path.join(log_dir, log_file)
        
        # Setup rotating file handler
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    else:
        # If no log file was specified, use a default name with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_path = os.path.join(log_dir, f"{app_name}_{timestamp}.log")
        
        # Setup rotating file handler
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Log the configuration
    root_logger.info(f"Logging configured with level={log_level}, file={log_path}")

def setup_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Setup and return a logger with the given name.
    
    Args:
        name: The logger name (usually module name)
        level: Optional specific log level for this logger
        
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    
    # Set specific level if provided
    if level is not None:
        numeric_level = getattr(logging, level.upper(), None)
        if numeric_level is not None:
            logger.setLevel(numeric_level)
    
    return logger

def set_log_level(level: str, logger_name: Optional[str] = None) -> None:
    """
    Set the log level for a logger or all loggers.
    
    Args:
        level: The log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        logger_name: The logger name (or None for root logger)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    if logger_name:
        logger = logging.getLogger(logger_name)
        logger.setLevel(numeric_level)
        logger.info(f"Log level for '{logger_name}' set to {level}")
    else:
        logging.getLogger().setLevel(numeric_level)
        logging.info(f"Global log level set to {level}")

# Initialize logging with default settings when module is imported
setup_logging(
    log_level=os.environ.get('LOG_LEVEL', 'INFO'),
    app_name='ai_detection',
    colored=True
)