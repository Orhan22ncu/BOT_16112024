import logging
import traceback
import sys
from datetime import datetime
import json
import threading
from collections import deque

class DebugTracker:
    def __init__(self, log_file='debug.log', max_memory=1000):
        self.logger = self._setup_logger(log_file)
        self.error_queue = deque(maxlen=max_memory)
        self.warning_queue = deque(maxlen=max_memory)
        self.lock = threading.Lock()
        
    def _setup_logger(self, log_file):
        logger = logging.getLogger('DebugTracker')
        logger.setLevel(logging.DEBUG)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def track_error(self, error, context=None):
        with self.lock:
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'type': type(error).__name__,
                'message': str(error),
                'traceback': traceback.format_exc(),
                'context': context
            }
            
            self.error_queue.append(error_info)
            self.logger.error(
                f"Error: {error_info['type']} - {error_info['message']}\n"
                f"Context: {context}\n"
                f"Traceback: {error_info['traceback']}"
            )
    
    def track_warning(self, message, context=None):
        with self.lock:
            warning_info = {
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'context': context
            }
            
            self.warning_queue.append(warning_info)
            self.logger.warning(
                f"Warning: {message}\n"
                f"Context: {context}"
            )
    
    def get_error_summary(self):
        with self.lock:
            return {
                'total_errors': len(self.error_queue),
                'recent_errors': list(self.error_queue)[-10:],
                'error_types': self._count_error_types()
            }
    
    def _count_error_types(self):
        error_types = {}
        for error in self.error_queue:
            error_type = error['type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
        return error_types
    
    def export_logs(self, filepath):
        with self.lock:
            with open(filepath, 'w') as f:
                json.dump({
                    'errors': list(self.error_queue),
                    'warnings': list(self.warning_queue)
                }, f, indent=2)
    
    def clear_logs(self):
        with self.lock:
            self.error_queue.clear()
            self.warning_queue.clear()