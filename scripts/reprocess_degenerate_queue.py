"""Cancel future degenerate scheduled posts + regenerate their clips' text.

The "viral" incident left hundreds of posts queued on Zernio whose whole
caption is a stray token (the old persona-name fallback) — e.g. literally
"viral\n\n@doomscroll" — scheduled days into the future. This script:

  1. finds the tenant's status='scheduled' posts whose scheduled_for is in
     the FUTURE and whose caption is degenerate (first block < 3 words);
  2. cancels each on Zernio and hard-deletes the local record;
  3. for each affected clip: regenerates a real hook (local LLM via
     generate_hook_line — deterministic fallback if the model is down),
     rewrites the variant row (hook + stream-title body) so every future
     compose produces real text, and re-stamps the overlay default;
  4. flips the clip back to 'approved' when no other live publish rows
     remain, so "Auto-programar todo" can re-schedule it.

Past/published rows are never touched — cancelling an already-published
post could delete the platform upload.

Run from the repo root with worker.env loaded (see
scripts/run_reprocess_degenerate.ps1):

    python scripts/reprocess_degenerate_queue.py --tenant ten_... [--live]

Default is DRY RUN: prints what would happen, changes nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os
import sys

from chalybclip.clip.overlay_defaults import default_overlay_config
from chalybclip.db import (
    ClipsRepo,
    Database,
    StreamsRepo,
    VariantsRepo,
    ZernioPublishesRepo,
)
from chalybclip.integrations.zernio.client import ZernioClient, ZernioError
from chalybclip.publish.compose import generate_hook_line
from chalybclip.tenancy import bound_tenant


def is_degenerate(content: str) -> bool:
    first_block = (content or "").strip().split("\n\n")[0].strip()
    words = [w for w in first_block.split() if not w.startswith(("#", "@"))]
    return len(words) < 3


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--live", action="store_true", help="Actually do it.")
    args = parser.parse_args()
    tenant_id: str = args.tenant
    live: bool = args.live

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
        clips_repo = ClipsRepo(db)
        with bound_tenant(tenant_id):
            rows = await pubs.list_for_tenant(limit=1000, status="scheduled")

        targets = [
            r for r in rows
            if (r.scheduled_for or "") > now_iso and is_degenerate(r.content or "")
        ]
        clip_ids: list[str] = []
        for r in targets:
            if r.clip_id and r.clip_id not in clip_ids:
                clip_ids.append(r.clip_id)
        print(f"scheduled={len(rows)} future+degenerate={len(targets)} "
              f"distinct_clips={len(clip_ids)} live={live}")
        if not live:
            for r in targets[:10]:
                print("would cancel:", r.post_id, r.clip_id, r.scheduled_for)
            print("... dry run, nothing changed. Re-run with --live.")
            return

        # 1) Cancel on Zernio + scrub local records.
        canceled = cancel_failed = 0
        for r in targets:
            try:
                try:
                    await client.delete_post(r.post_id)
                except ZernioError as e:
                    print(f"  zernio delete failed (continuing): {r.post_id}: {e}")
                await pubs.delete(r.post_id)
                canceled += 1
            except Exception as e:
                cancel_failed += 1
                print(f"  CANCEL FAILED {r.post_id}: {e}")
        print(f"canceled={canceled} cancel_failed={cancel_failed}")

        # 2) Fix each clip's text + overlay; re-approve when fully cleared.
        fixed = reapproved = 0
        for i, clip_id in enumerate(clip_ids, 1):
            try:
                with bound_tenant(tenant_id):
                    clip = await clips_repo.get(clip_id)
                    if clip is None:
                        continue
                    stream = await StreamsRepo(db).get(clip.stream_id)
                stream_title = (stream.title or "").strip() if stream else ""

                hook = await generate_hook_line(db, tenant_id, clip_id)
                body = stream_title or hook

                with bound_tenant(tenant_id):
                    variants = await VariantsRepo(db).list_for_clip(clip_id)
                    for v in variants:
                        v.title_card_text = hook
                        if is_degenerate(v.caption or ""):
                            v.caption = body[:240]
                    by_persona: dict[str, list] = {}
                    for v in variants:
                        by_persona.setdefault(v.persona_id, []).append(v)
                    for persona_id, batch in by_persona.items():
                        await VariantsRepo(db).replace_for_clip_persona(
                            clip_id, persona_id, batch
                        )
                    await clips_repo.set_overlay_config(
                        clip_id,
                        overlay_config=default_overlay_config(
                            platform=getattr(stream, "platform", None) if stream else None,
                            channel=getattr(stream, "channel", None) if stream else None,
                            hook=hook,
                        ),
                    )
                    if not await pubs.exists_for_clip(tenant_id, clip_id):
                        await clips_repo.update_status(clip_id, status="approved")
                        reapproved += 1
                fixed += 1
                print(f"[{i}/{len(clip_ids)}] {clip_id} hook={hook[:60]!r}")
            except Exception as e:
                print(f"  CLIP FIX FAILED {clip_id}: {e}")
        print(f"clips_fixed={fixed} reapproved={reapproved}")
    finally:
        await db.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
