"""
Logging configuration for QuantLab application.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


class QuantLabLogger:
    """Custom logger for QuantLab with proper formatting and handlers."""
    
    def __init__(self, name: str = "quantlab", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up console and file handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # File handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "quantlab.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance for the given name."""
    if name is None:
        name = "quantlab"
    
    logger_instance = QuantLabLogger(name)
    return logger_instance.get_logger()


# Module-level logger for convenience
logger = get_logger()