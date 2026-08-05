from celery import Celery

from app.core.config import get_settings

settings = get_settings()
celery_app = Celery("caspian_guardian", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    beat_schedule={
        "discover-new-scenes": {
            "task": "guardian.monitor_due_areas",
            "schedule": float(settings.monitoring_interval_minutes * 60),
        }
    },
)


@celery_app.task(
    name="guardian.run_analysis",
    bind=True,
    autoretry_for=(RuntimeError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 4},
)
def run_analysis_task(self, job_id: str) -> None:
    import asyncio
    from uuid import UUID

    from app.main import run_analysis

    asyncio.run(run_analysis(UUID(job_id)))


@celery_app.task(
    name="guardian.monitor_due_areas",
    bind=True,
    autoretry_for=(RuntimeError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def monitor_due_areas_task(self) -> int:
    import asyncio

    from app.main import monitor_due_areas

    return asyncio.run(monitor_due_areas())
