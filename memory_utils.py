"""
Memory management utilities for document processing applications.
Simple yet effective memory optimization tools.
"""

import gc
import sys
import logging
import time
from functools import wraps

# Configure logger
logger = logging.getLogger(__name__)

def get_memory_usage():
    """
    Get current memory usage of the process in MB.
    Returns tuple of (rss, vms) memory in MB.
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return (memory_info.rss / (1024 * 1024), memory_info.vms / (1024 * 1024))
    except ImportError:
        logger.warning("psutil not installed, can't report memory usage")
        return (0, 0)
    except Exception as e:
        logger.warning(f"Error getting memory usage: {str(e)}")
        return (0, 0)

def cleanup_memory(log_label=""):
    """
    Simple but effective memory cleanup.
    
    Args:
        log_label (str): Optional label for logging
    
    Returns:
        tuple: Memory before and after cleanup in MB, or (0,0) if monitoring not available
    """
    label = f"[{log_label}] " if log_label else ""
    
    # Get memory before cleanup
    mem_before, _ = get_memory_usage()
    if mem_before > 0:
        logger.info(f"{label}Memory before cleanup: {mem_before:.2f} MB")
    
    # Run multiple garbage collection cycles for thoroughness
    collected = 0
    for i in range(3):
        collected += gc.collect()
    
    logger.info(f"{label}Memory cleanup: Collected {collected} objects")
    
    # Try to release memory to the OS (Linux only)
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6')
        if hasattr(libc, 'malloc_trim'):
            result = libc.malloc_trim(0)
            if result > 0:
                logger.info(f"{label}Successfully released memory to OS via malloc_trim")
    except Exception:
        pass  # Silently continue if this fails
    
    # Get memory after cleanup
    time.sleep(0.1)  # Short delay to allow OS to reclaim memory
    mem_after, _ = get_memory_usage()
    
    if mem_before > 0 and mem_after > 0:
        diff = mem_before - mem_after
        logger.info(f"{label}Memory after cleanup: {mem_after:.2f} MB (reduced by {diff:.2f} MB)")
    
    return (mem_before, mem_after)

def memory_managed(func=None, label=None, log_memory=True):
    """
    Decorator to add memory management to functions.
    
    Args:
        func: The function to decorate
        label: Optional label for logging (defaults to function name)
        log_memory: Whether to log memory usage
        
    Usage:
        @memory_managed
        def my_function():
            # ...
        
        # Or with parameters:
        @memory_managed(label="PDF Processing", log_memory=True)
        def process_pdf():
            # ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            func_name = label or fn.__name__
            
            if log_memory:
                logger.info(f"Starting memory-managed function: {func_name}")
            
            try:
                # Execute the function
                result = fn(*args, **kwargs)
                
                # Clean up memory after function execution
                cleanup_memory(func_name)
                
                return result
                
            except Exception as e:
                # Always clean up on error
                logger.error(f"Error in {func_name}: {str(e)}")
                cleanup_memory(f"{func_name}:error")
                raise
        
        return wrapper
    
    # Handle both @memory_managed and @memory_managed()
    if func is None:
        return decorator
    return decorator(func)

def clean_variables(*variables):
    """
    Explicitly delete variables and run garbage collection.
    
    Args:
        *variables: Variables to delete
        
    Example:
        image_data = load_large_image()
        # Process image...
        clean_variables(image_data)
    """
    for var in variables:
        if var is not None:
            var_type = type(var).__name__
            try:
                var_size = sys.getsizeof(var) / (1024 * 1024)
                logger.debug(f"Deleting {var_type} object (approx. {var_size:.2f} MB)")
            except:
                logger.debug(f"Deleting {var_type} object")
            
            del var
    
    # Run quick garbage collection to reclaim memory
    gc.collect(0)  # Quick collection of youngest generation only

# Initialize module
logger.info("Memory management utilities loaded")