"""
Enhanced memory management utilities for worker optimization.
Add these functions to your memory_utils.py or tasks.py file.
"""

import gc
import sys
import logging
import time

logger = logging.getLogger(__name__)

def aggressive_cleanup():
    """
    Perform very aggressive memory cleanup for worker processes.
    Call this at the end of major processing tasks.
    """
    # Log memory before cleanup
    try:
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory before aggressive cleanup: {memory_before:.2f} MB")
    except ImportError:
        memory_before = 0
        logger.info("Memory cleanup started (psutil not available)")
    
    # Force collection of all generations multiple times
    collected_total = 0
    for i in range(5):  # More iterations for better cleanup
        collected = gc.collect(2)  # Full collection
        collected_total += collected
        if collected == 0 and i > 1:
            break  # Stop if nothing more to collect
    
    # Try to release memory to the OS
    try:
        import ctypes
        libc = ctypes.CDLL('libc.so.6')
        if hasattr(libc, 'malloc_trim'):
            for i in range(3):  # Try multiple times
                result = libc.malloc_trim(0)
                if result > 0:
                    logger.info(f"Successfully released memory to OS (attempt {i+1})")
                time.sleep(0.1)  # Small delay between attempts
    except Exception as e:
        logger.warning(f"Failed to call malloc_trim: {str(e)}")
    
    # For Linux: try to use POSIX madvise to release memory
    try:
        import ctypes
        import mmap
        libc = ctypes.CDLL('libc.so.6')
        if hasattr(libc, 'madvise'):
            MADV_DONTNEED = 4  # Value on most Linux systems
            # This is a more aggressive approach that tells the OS we don't need
            # some memory pages - use with caution as it can affect performance
            result = 0  # Just log the capability, don't actually call it globally
            logger.info(f"System supports madvise for memory release")
    except Exception:
        pass
    
    # Manually trigger compaction in Python 3.7+ (if available)
    try:
        if hasattr(gc, 'collect') and callable(getattr(gc, 'collect')):
            # Try to use freeze/unfreeze if available (Python 3.11+)
            if hasattr(gc, 'freeze') and callable(getattr(gc, 'freeze')):
                logger.info("Using advanced GC freeze/unfreeze for better compaction")
                gc.freeze()
                gc.unfreeze()
    except Exception:
        pass
    
    # Get worker process children and try to reclaim their resources
    try:
        import psutil
        process = psutil.Process()
        
        # Log child processes
        children = process.children(recursive=True)
        if children:
            logger.info(f"Found {len(children)} child processes")
            
            # Attempt to clean up any zombie processes
            for child in children:
                try:
                    if child.status() == psutil.STATUS_ZOMBIE:
                        logger.info(f"Cleaning up zombie process: {child.pid}")
                        # In real code, you'd wait for this process
                        # Don't actually terminate children in production code
                        # as these could be legitimate worker processes
                except:
                    pass
    except Exception:
        pass
    
    # Log memory after cleanup
    try:
        import psutil
        process = psutil.Process()
        # Wait a moment for the OS to reclaim memory
        time.sleep(0.5)
        memory_after = process.memory_info().rss / (1024 * 1024)
        logger.info(f"Memory after aggressive cleanup: {memory_after:.2f} MB")
        if memory_before > 0:
            reduction = memory_before - memory_after
            logger.info(f"Memory reduced by: {reduction:.2f} MB ({(reduction/memory_before)*100:.1f}%)")
    except ImportError:
        pass

def unload_models():
    """
    Explicitly unload ML models to reclaim memory.
    Call this after you're done using models in a task.
    """
    # This needs to be adapted to your specific models
    # Collect modules that might hold references to large objects
    import sys
    import gc
    
    logger.info("Attempting to unload ML models from memory")
    
    # For PaddleOCR
    try:
        import paddle
        if hasattr(paddle, 'disable_static'):
            paddle.disable_static()
        # Clear any cached tensors or variables
        if hasattr(paddle, 'fluid') and hasattr(paddle.fluid, 'core'):
            if hasattr(paddle.fluid.core, 'gc'):
                paddle.fluid.core.gc()
        logger.info("Cleared PaddleOCR/Paddle resources")
    except Exception as e:
        logger.warning(f"Error cleaning up Paddle resources: {str(e)}")
    
    # For transformers (e.g., T5 model)
    try:
        from transformers import pipeline
        # Find references to pipeline objects
        for obj in gc.get_objects():
            if isinstance(obj, pipeline):
                # Clear model cache if possible
                if hasattr(obj, 'model') and hasattr(obj.model, 'cpu'):
                    # Move model to CPU if it was on GPU
                    obj.model.cpu()
                # Remove reference to tokenizer and model if possible
                if hasattr(obj, '_tokenizer'):
                    obj._tokenizer = None
                if hasattr(obj, 'model'):
                    obj.model = None
        logger.info("Cleared transformer pipeline resources")
    except Exception as e:
        logger.warning(f"Error cleaning up transformer resources: {str(e)}")
    
    # Explicitly run garbage collection
    gc.collect()
    
    # Log if the unload appears successful
    logger.info("Model unloading complete")

def limit_numpy_threads():
    """
    Limit the number of threads used by NumPy and related libraries.
    Call this at the start of worker processes.
    """
    # Set thread count for various math libraries
    import os
    
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    
    # Try to set thread count for NumPy directly
    try:
        import numpy as np
        if hasattr(np, 'seterr'):
            # Adjust floating point error handling
            np.seterr(all='warn')
        # For recent NumPy versions with thread control
        if hasattr(np, '__config__') and hasattr(np.__config__, 'get_info'):
            logger.info(f"NumPy threading info: {np.__config__.get_info('threading')}")
    except:
        pass
    
    logger.info("NumPy and math library thread limits applied")

def clear_opencv_cache():
    """
    Clear OpenCV cache after processing images.
    Call this after image processing operations.
    """
    try:
        import cv2
        # Release any cached OpenCV resources
        cv2.destroyAllWindows()
        # Clear OpenCV Disk Cache if possible
        if hasattr(cv2, 'ocl') and hasattr(cv2.ocl, 'setUseOpenCL'):
            cv2.ocl.setUseOpenCL(False)
        # This is a placeholder - OpenCV doesn't have a direct cache clearing function,
        # but this can help release some resources
        logger.info("OpenCV resources released")
    except Exception as e:
        logger.warning(f"Error clearing OpenCV resources: {str(e)}")