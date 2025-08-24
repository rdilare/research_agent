"""
Debug and logging helper functions
"""
import logging
import json
import time
from typing import Any, Dict
from datetime import datetime


class DebugLogger:
    """Enhanced logging for debugging research agent"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.debug_mode = config.get('logging', {}).get('debug_mode', False)
        self.step_logging = config.get('logging', {}).get('step_logging', True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configure logging format
        self._configure_logging()
    
    def _configure_logging(self):
        """Configure logging format and level"""
        log_level = self.config.get('logging', {}).get('level', 'INFO')
        
        if self.debug_mode:
            log_level = 'DEBUG'
        
        # Create formatter
        if self.config.get('logging', {}).get('format', 'json') == 'json':
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        else:
            formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            )
        
        # Configure handlers
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.setLevel(getattr(logging, log_level.upper()))
    
    def log_step_start(self, step_name: str, inputs: Any = None):
        """Log the start of a processing step"""
        if not self.step_logging:
            return
        
        self.logger.info(f"Starting step: {step_name}")
        
        if self.debug_mode and inputs is not None:
            self.logger.debug(f"Step '{step_name}' inputs: {self._safe_serialize(inputs)}")
    
    def log_step_end(self, step_name: str, outputs: Any = None, execution_time: float = 0):
        """Log the end of a processing step"""
        if not self.step_logging:
            return
        
        self.logger.info(f"Completed step: {step_name} (took {execution_time:.2f}s)")
        
        if self.debug_mode and outputs is not None:
            self.logger.debug(f"Step '{step_name}' outputs: {self._safe_serialize(outputs)}")
    
    def log_step_error(self, step_name: str, error: Exception):
        """Log an error in a processing step"""
        self.logger.error(f"Error in step '{step_name}': {str(error)}")
        
        if self.debug_mode:
            import traceback
            self.logger.debug(f"Full traceback for '{step_name}': {traceback.format_exc()}")
    
    def _safe_serialize(self, data: Any, max_length: int = 1000) -> str:
        """Safely serialize data for logging"""
        try:
            if isinstance(data, (dict, list)):
                serialized = json.dumps(data, indent=2)
            else:
                serialized = str(data)
            
            # Truncate if too long
            if len(serialized) > max_length:
                serialized = serialized[:max_length] + "... (truncated)"
            
            return serialized
        except Exception:
            return f"<Unable to serialize {type(data).__name__}>"


def create_step_tracker():
    """Create a context manager for tracking step execution"""
    
    class StepTracker:
        def __init__(self, step_name: str, logger: DebugLogger):
            self.step_name = step_name
            self.logger = logger
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            self.logger.log_step_start(self.step_name)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            execution_time = time.time() - self.start_time
            
            if exc_type is None:
                self.logger.log_step_end(self.step_name, execution_time=execution_time)
            else:
                self.logger.log_step_error(self.step_name, exc_val)
    
    return StepTracker


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and enhance configuration for debugging"""
    
    # Ensure logging section exists
    if 'logging' not in config:
        config['logging'] = {}
    
    # Set defaults
    logging_config = config['logging']
    logging_config.setdefault('level', 'INFO')
    logging_config.setdefault('debug_mode', False)
    logging_config.setdefault('step_logging', True)
    logging_config.setdefault('format', 'json')
    logging_config.setdefault('max_log_size', '10MB')
    logging_config.setdefault('backup_count', 5)
    
    return config


def create_debug_summary(results: Any) -> Dict[str, Any]:
    """Create a summary for debug display"""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'has_step_logs': hasattr(results, 'step_logs'),
        'has_execution_times': hasattr(results, 'execution_times'),
        'has_errors': hasattr(results, 'errors') and len(results.errors) > 0,
        'total_steps': 0,
        'total_execution_time': 0,
        'error_count': 0
    }
    
    if hasattr(results, 'step_logs'):
        summary['total_steps'] = len(results.step_logs)
        summary['steps'] = [log['step'] for log in results.step_logs]
    
    if hasattr(results, 'execution_times'):
        summary['total_execution_time'] = sum(results.execution_times.values())
        summary['slowest_step'] = max(results.execution_times.items(), key=lambda x: x[1]) if results.execution_times else None
        summary['fastest_step'] = min(results.execution_times.items(), key=lambda x: x[1]) if results.execution_times else None
    
    if hasattr(results, 'errors'):
        summary['error_count'] = len(results.errors)
        summary['errors'] = results.errors
    
    return summary


def format_debug_output(data: Any) -> str:
    """Format data for debug display"""
    if isinstance(data, dict):
        try:
            return json.dumps(data, indent=2)
        except:
            return str(data)
    elif isinstance(data, list):
        if len(data) <= 5:
            return json.dumps(data, indent=2)
        else:
            return f"List with {len(data)} items:\n{json.dumps(data[:3], indent=2)}\n... (showing first 3 items)"
    else:
        return str(data)
