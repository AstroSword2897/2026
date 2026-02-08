"""Structured logging for MaxSight Web Simulator. Component-based logging with consistent format."""
import logging
import json
from typing import Any, Dict, Optional
from datetime import datetime
from .config import config


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'component': getattr(record, 'component', 'unknown'),
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present.
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields.
        if hasattr(record, 'session_id'):
            log_data['session_id'] = record.session_id
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        # Add any extra fields.
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName', 'relativeCreated',
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info',
                          'component', 'session_id', 'user_id', 'request_id']:
                log_data[key] = value
        
        return json.dumps(log_data) if config.enable_structured_logging else self._format_text(log_data)
    
    def _format_text(self, log_data: Dict[str, Any]) -> str:
        """Fallback text format."""
        timestamp = log_data.get('timestamp', '')
        level = log_data.get('level', 'INFO')
        component = log_data.get('component', 'unknown')
        message = log_data.get('message', '')
        return f"{timestamp} [{level}] [{component}] {message}"


class ComponentLogger:
    """Logger wrapper with component context."""
    
    def __init__(self, component: str, base_logger: logging.Logger):
        self.component = component
        self.logger = base_logger
    
    def _log(self, level: int, msg: str, *args, session_id: Optional[str] = None,
             user_id: Optional[str] = None, request_id: Optional[str] = None,
             **kwargs):
        """Internal logging method with component context."""
        extra = {
            'component': self.component,
            **kwargs
        }
        if session_id:
            extra['session_id'] = session_id
        if user_id:
            extra['user_id'] = user_id
        if request_id:
            extra['request_id'] = request_id
        
        self.logger.log(level, msg, *args, extra=extra, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, exc_info=True, **kwargs):
        """Log exception."""
        self._log(logging.ERROR, msg, *args, exc_info=exc_info, **kwargs)


def setup_structured_logging(log_level: str = None) -> logging.Logger:
    """Setup structured logging for the simulator. Args: log_level: Logging level (uses config if None) Returns: Configured logger."""
    if log_level is None:
        log_level = config.log_level
    
    logger = logging.getLogger('maxsight_simulator')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Remove existing handlers.
    logger.handlers.clear()
    
    # Create console handler.
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(StructuredFormatter())
    
    logger.addHandler(handler)
    logger.propagate = False
    
    return logger


def get_component_logger(component: str) -> ComponentLogger:
    """Get logger for a specific component. Args: component: Component name (e.g., 'session', 'api', 'core') Returns: ComponentLogger instance."""
    base_logger = logging.getLogger('maxsight_simulator')
    return ComponentLogger(component, base_logger)







