"""Google Drive folder watches — voice-markers spec slice E.4.

The automation path: a streamer connects a Drive folder once; ChalyClip
polls it on a schedule and auto-ingests any new VOD files that land.

Layered like the rest of the codebase:

  * `models.py`   — Pydantic types (`DriveFile`, `DriveWatchRow`)
  * `protocol.py` — `DriveClient` protocol + `FakeDriveClient` for tests
                    / local dev (no Google API dep required)
  * `service.py`  — `poll_drive_watches(db, drive_client_factory,
                    ingest_callback)` — the actual polling loop

The Google API client itself (google-api-python-client + google-auth)
is an optional extra (`pip install 'chalybclip[drive]'`). Like pyannote,
the dashboard ships without it — `drive.is_google_client_available()`
short-circuits the polling loop when the import fails.

Production wiring (deferred to a follow-up commit):
  * `chalybclip/drive/google.py` — real `GoogleDriveClient` implementing
    the protocol, OAuth refresh, `files.list` + `files.export_media`.
  * `chalybclip/drive/oauth.py` — OAuth handshake routes on the dashboard.
"""

from __future__ import annotations

from .models import DriveFile, DriveWatchRow
from .protocol import DriveClient, FakeDriveClient
from .service import PollReport, poll_drive_watches

__all__ = [
    "DriveClient",
    "DriveFile",
    "DriveWatchRow",
    "FakeDriveClient",
    "PollReport",
    "poll_drive_watches",
]
