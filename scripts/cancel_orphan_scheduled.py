"""Cancel FUTURE scheduled Zernio posts whose source clip no longer exists.

A post whose clip row is gone can never publish correctly: the media URL
404s at slot time and surfaces as a "post failed" email (the failure mode
the stream-delete guard now prevents at the source). This scrubs the ones
that predate the guard: cancel on Zernio + hard-delete the local record.

    python scripts/cancel_orphan_scheduled.py --tenant ten_... [--live]

Dry run by default. Requires worker.env loaded (DATABASE_URL +
CHALYBCLIP_ZERNIO_API_KEY) — see scripts/run_reprocess_degenerate.ps1 for
the loading pattern.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import sys

from chalybclip.db import ClipsRepo, Database, ZernioPublishesRepo
from chalybclip.integrations.zernio.client import ZernioClient, ZernioError
from chalybclip.tenancy import bound_tenant


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()

    db = Database(os.environ["DATABASE_URL"])
    await db.connect()
    client = ZernioClient(
        api_key=os.environ["CHALYBCLIP_ZERNIO_API_KEY"],
        base_url=os.environ.get("CHALYBCLIP_ZERNIO_BASE_URL")
        or "https://zernio.com/api/v1",
    )
    now_iso = dt.datetime.now(dt.UTC).isoformat()

    try:
        pubs = ZernioPublishesRepo(db)
        clips = ClipsRepo(db)
        with bound_tenant(args.tenant):
            rows = await pubs.list_for_tenant(limit=1000, status="scheduled")
            orphans = []
            for r in rows:
                if (r.scheduled_for or "") <= now_iso:
                    continue
                if r.clip_id and await clips.get(r.clip_id) is not None:
                    continue
                orphans.append(r)

        print(f"scheduled={len(rows)} future_orphans={len(orphans)} live={args.live}")
        if not args.live:
            for r in orphans[:10]:
                print("would cancel:", r.post_id, r.clip_id, r.scheduled_for)
            print("... dry run, nothing changed. Re-run with --live.")
            return

        canceled = failed = 0
        for r in orphans:
            try:
                try:
                    await client.delete_post(r.post_id)
                except ZernioError as e:
                    print(f"  zernio delete failed (continuing): {r.post_id}: {e}")
                await pubs.delete(r.post_id)
                canceled += 1
            except Exception as e:
                failed += 1
                print(f"  CANCEL FAILED {r.post_id}: {e}")
        print(f"canceled={canceled} failed={failed}")
    finally:
        await db.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
