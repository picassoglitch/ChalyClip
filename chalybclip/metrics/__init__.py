"""Engagement-metrics ingestion.

Per-platform fetchers pull view / like / comment / retention numbers for
each `sent` publish_job's `external_id`. The ingest service drains all
eligible jobs for a tenant and writes one snapshot row per fetch through
the PublishMetricsRepo. The dashboard's outcome card reads the latest
row; the calibration loop reads the time series.

Two callers:
    * `chalybclip metrics ingest --tenant <id>` for one-shot drains.
    * The FastAPI lifespan task wakes hourly and runs the same drain
      (Phase 3 worker pool will move this to ECS).
"""

from .calibration import CalibrationReport, CalibrationRow, compute_calibration
from .fetchers import (
    FetcherProtocol,
    NormalizedMetric,
    fetch_buffer_metric,
    fetch_instagram_metric,
    fetch_tiktok_metric,
    fetch_youtube_metric,
)
from .service import IngestOutcome, run_metrics_ingest

__all__ = [
    "CalibrationReport",
    "CalibrationRow",
    "FetcherProtocol",
    "IngestOutcome",
    "NormalizedMetric",
    "compute_calibration",
    "fetch_buffer_metric",
    "fetch_instagram_metric",
    "fetch_tiktok_metric",
    "fetch_youtube_metric",
    "run_metrics_ingest",
]
