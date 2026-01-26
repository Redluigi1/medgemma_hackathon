"""
LLM Logger Module
Comprehensive logging for all LLM calls with timestamps, prompts, responses, and durations.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from functools import wraps


class LLMLogger:
    """Logger for tracking all LLM API calls."""
    
    def __init__(self, log_dir: str = None):
        if log_dir is None:
            log_dir = Path(__file__).parent / "logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup file logger
        self.logger = logging.getLogger("LLMLogger")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for text logs
        log_file = self.log_dir / "llm_calls.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        # JSON log file for structured data
        self.json_log_file = self.log_dir / "llm_calls.json"
        self.call_history = []
        self._load_existing_logs()
    
    def _load_existing_logs(self):
        """Load existing JSON logs if they exist."""
        if self.json_log_file.exists():
            try:
                with open(self.json_log_file, 'r', encoding='utf-8') as f:
                    self.call_history = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.call_history = []
    
    def _save_json_log(self):
        """Save call history to JSON file."""
        with open(self.json_log_file, 'w', encoding='utf-8') as f:
            json.dump(self.call_history, f, indent=2, ensure_ascii=False)
    
    def log_call(self, 
                 function_name: str,
                 prompt: str,
                 image_paths: list = None,
                 response: str = None,
                 duration_seconds: float = None,
                 success: bool = True,
                 error: str = None,
                 metadata: dict = None):
        """
        Log an LLM API call.
        
        Args:
            function_name: Name of the function making the call
            prompt: The text prompt sent to the LLM
            image_paths: List of image paths if multimodal
            response: The LLM response
            duration_seconds: Time taken for the call
            success: Whether the call was successful
            error: Error message if failed
            metadata: Additional metadata
        """
        timestamp = datetime.now().isoformat()
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "function": function_name,
            "prompt": prompt[:500] + "..." if len(prompt) > 500 else prompt,
            "prompt_length": len(prompt),
            "image_paths": image_paths or [],
            "num_images": len(image_paths) if image_paths else 0,
            "response": response[:1000] + "..." if response and len(response) > 1000 else response,
            "response_length": len(response) if response else 0,
            "duration_seconds": round(duration_seconds, 3) if duration_seconds else None,
            "success": success,
            "error": error,
            "metadata": metadata or {}
        }
        
        # Log to text file
        if success:
            self.logger.info(
                f"[{function_name}] Prompt: {len(prompt)} chars | "
                f"Images: {len(image_paths) if image_paths else 0} | "
                f"Response: {len(response) if response else 0} chars | "
                f"Duration: {duration_seconds:.2f}s" if duration_seconds else ""
            )
        else:
            self.logger.error(f"[{function_name}] FAILED: {error}")
        
        # Add to history and save
        self.call_history.append(log_entry)
        self._save_json_log()
        
        return log_entry
    
    def get_summary(self):
        """Get summary statistics of all LLM calls."""
        if not self.call_history:
            return {"total_calls": 0}
        
        successful = [c for c in self.call_history if c["success"]]
        failed = [c for c in self.call_history if not c["success"]]
        
        total_duration = sum(c["duration_seconds"] or 0 for c in successful)
        
        return {
            "total_calls": len(self.call_history),
            "successful": len(successful),
            "failed": len(failed),
            "total_duration_seconds": round(total_duration, 2),
            "avg_duration_seconds": round(total_duration / len(successful), 2) if successful else 0,
            "total_images_processed": sum(c["num_images"] for c in self.call_history)
        }


# Global logger instance
_logger = None

def get_logger(log_dir: str = None) -> LLMLogger:
    """Get or create the global LLM logger instance."""
    global _logger
    if _logger is None:
        _logger = LLMLogger(log_dir)
    return _logger


def logged_llm_call(func):
    """Decorator to automatically log LLM calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        start_time = time.time()
        
        # Extract prompt and image paths from args/kwargs
        prompt = kwargs.get('prompt', args[0] if args else "")
        image_paths = kwargs.get('image_paths', args[1] if len(args) > 1 else None)
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            # Extract response text
            response_text = None
            if result:
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        response_text = result[0].get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', str(result))
                    else:
                        response_text = str(result)
                else:
                    response_text = str(result)
            
            logger.log_call(
                function_name=func.__name__,
                prompt=prompt if isinstance(prompt, str) else str(prompt),
                image_paths=image_paths if isinstance(image_paths, list) else None,
                response=response_text,
                duration_seconds=duration,
                success=True
            )
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.log_call(
                function_name=func.__name__,
                prompt=prompt if isinstance(prompt, str) else str(prompt),
                image_paths=image_paths if isinstance(image_paths, list) else None,
                duration_seconds=duration,
                success=False,
                error=str(e)
            )
            raise
    
    return wrapper