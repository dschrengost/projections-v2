"""Background job helpers for projections-v2.

This package intentionally avoids introducing a full task queue (Celery/RQ).
Jobs are executed as subprocesses with on-disk status/logs for UI polling.
"""

