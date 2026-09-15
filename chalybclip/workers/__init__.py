"""Self-hosted workers — pipeline compute off the web box, off the cloud bill.

`pc_worker` is the operator-hardware twin of `infra/modal_pipeline_app.py`:
the same kickoff/poll HTTP contract, running on a machine the operator owns
(GPU box), against the same shared Postgres + R2.
"""

from .pc_worker import create_worker_app

__all__ = ["create_worker_app"]
