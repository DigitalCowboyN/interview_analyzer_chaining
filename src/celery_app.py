"""
src/celery_app.py

Configuration and entry point for the Celery application instance.
"""

import os
from celery import Celery

# TODO: Load broker URL from config/env vars later
# Defaulting to standard Redis port, service name 'redis'
redis_url = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")

celery_app = Celery(
    "interview_analyzer", # Name of the project/tasks module
    broker=redis_url,
    backend=redis_url, # Using Redis as backend too for now (optional)
    include=["src.tasks"]  # List of modules where tasks are defined (adjust later if needed)
)

# Optional configuration settings (can be moved to config.py)
celery_app.conf.update(
    result_expires=3600, # Keep results for 1 hour (example)
    task_serializer='json',
    accept_content=['json'],  # Ensure tasks use json
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Optional: Load configuration from a central config object if preferred later
# celery_app.config_from_object('src.config.celeryconfig') 

if __name__ == "__main__":
    celery_app.start() # Allows running worker directly via `python src/celery_app.py worker` 