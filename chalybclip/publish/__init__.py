"""Publish package.

Two live surfaces, both imported by full path (no package-level re-exports):

* `chalybclip.publish.hub` — the Zernio hub publish API: resolve targets,
  build per-platform payloads, create/track HubPublishJobs. Used by the
  dashboard publish flow and the internal publish API.
* `chalybclip.publish.analytics_service` — per-tenant performance snapshots,
  weekly digests, and internal analytics reads.

The legacy per-platform worker — the `publish_jobs` drain (`run_publish_jobs`),
the Buffer/TikTok/YouTube/Instagram adapters, the OAuth refresh helper, and
the auto-publish dispatcher — was removed. Publishing goes through Zernio.
"""
