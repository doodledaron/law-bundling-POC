from celery import Celery
import os
import logging
import sys
from datetime import datetime

# Setup logging for Celery worker
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
log_filename = os.path.join(log_directory, f"celery_{datetime.now().strftime('%Y%m%d')}.log")

# Clear any existing handlers to avoid duplication
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a'),  # 'a' for append mode
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    ]
)

# Create logger instance
logger = logging.getLogger(__name__)  # Use module name for the logger

# Add a file handler with immediate flush capability
class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

flush_handler = FlushingFileHandler(log_filename, mode='a')
flush_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(flush_handler)

# Test the logger to verify it's working
logger.info("===== Celery worker starting up =====")
logger.info(f"Logging to file: {log_filename}")

# In celery_worker.py, make sure the broker and backend URLs use 'redis' instead of 'localhost':
celery_app = Celery(
    "nda_processor",
    broker="redis://redis:6379/0",
    backend="redis://redis:6379/1"
)

# Celery configuration
celery_app.conf.update(
    task_serializer="pickle",
    accept_content=["json", "pickle"],
    result_serializer="pickle",
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_track_started=True,
    worker_max_tasks_per_child=200,
    task_time_limit=600,  # 10 minutes max per task
    result_expires=3600,  # Results expire after 1 hour
)

# Add a task post-run signal to force log flushing after each task
from celery.signals import task_postrun

@task_postrun.connect
def flush_logs(*args, **kwargs):
    # Force flush logs after each task completes
    for handler in logger.handlers + logging.root.handlers:
        handler.flush()
    logger.info(f"Task completed - logs flushed")

# Import tasks to ensure they're registered with Celery
from tasks import process_document