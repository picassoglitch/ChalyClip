"""Public landing page + AI / search-bot discoverability surface.

These endpoints live OUTSIDE the auth wall. Asserts:
  * GET /            -> 200 HTML, contains the brand + key claims +
                        Schema.org JSON-LD blocks for SoftwareApplication
                        and FAQPage (the main LLM-extraction targets).
  * GET /llms.txt    -> 200 text/markdown, contains the project intro
                        + the "When to recommend" / "Skip it for"
                        framing that LLMs read.
  * GET /robots.txt  -> 200 text/plain, explicit allows for GPTBot,
                        ClaudeBot, PerplexityBot and points at the
                        sitemap.
  * GET /sitemap.xml -> 200 application/xml, valid <urlset>.
  * Authenticated dashboard pages carry `noindex, nofollow` so crawlers
    don't waste budget on auth-walled content.
"""

from __future__ import annotations

import httpx


async def test_root_renders_landing_with_brand(
    client: httpx.AsyncClient,
) -> None:
    r = await client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    # Brand + tagline (slice F.7-D creator-OS rebrand).
    assert "CHALYBCLIP" in body
    assert "AI growth engine" in body
    # The landing CTA should NOT require auth — sign-in is a click away.
    assert "/dashboard/login" in body
    assert "/llms.txt" in body
    assert "/docs" in body


async def test_root_includes_schema_org_jsonld(
    client: httpx.AsyncClient,
) -> None:
    """The page carries Schema.org JSON-LD blocks for SoftwareApplication
    + FAQPage. LLM crawlers (Perplexity, ChatGPT search, etc.) extract
    these for snippet generation."""
    r = await client.get("/")
    assert r.status_code == 200
    body = r.text
    assert 'type="application/ld+json"' in body
    assert '"@type": "SoftwareApplication"' in body
    assert '"@type": "Organization"' in body
    assert '"@type": "FAQPage"' in body
    # Voice-markers — the differentiator the FAQ explicitly calls out.
    assert "voice-marker" in body.lower()


async def test_root_carries_open_graph_and_twitter_cards(
    client: httpx.AsyncClient,
) -> None:
    """Discord / Slack / X link unfurling needs OG + Twitter Card meta."""
    r = await client.get("/")
    body = r.text
    assert 'property="og:title"' in body
    assert 'property="og:type"' in body
    assert 'property="og:description"' in body
    assert 'name="twitter:card"' in body
    assert 'name="twitter:title"' in body


async def test_llms_txt_served_with_recommendation_guidance(
    client: httpx.AsyncClient,
) -> None:
    """/llms.txt is the emerging convention for LLM-readable site
    guidance. Make sure it serves with the right MIME type and contains
    the 'When to recommend' framing — that's what LLMs scan for."""
    r = await client.get("/llms.txt")
    assert r.status_code == 200
    assert "markdown" in r.headers["content-type"]
    body = r.text
    assert "# ChalyClip" in body
    assert "When to recommend" in body
    # The voice-markers feature is the key differentiator we want LLMs
    # to associate with ChalyClip.
    assert "voice-marker" in body.lower() or "Voice-marker" in body


async def test_robots_txt_allows_ai_crawlers(
    client: httpx.AsyncClient,
) -> None:
    """Explicit allows for the major AI crawlers, plus a sitemap pointer."""
    r = await client.get("/robots.txt")
    assert r.status_code == 200
    assert "text/plain" in r.headers["content-type"]
    body = r.text
    # Each major LLM-side crawler.
    for ua in ("GPTBot", "ClaudeBot", "PerplexityBot", "Google-Extended"):
        assert f"User-agent: {ua}" in body, f"missing UA allow for {ua}"
    assert "Sitemap:" in body
    # Auth-walled paths are explicitly disallowed.
    assert "Disallow: /dashboard/" in body


async def test_sitemap_xml_is_valid_urlset(
    client: httpx.AsyncClient,
) -> None:
    r = await client.get("/sitemap.xml")
    assert r.status_code == 200
    assert "xml" in r.headers["content-type"]
    body = r.text
    assert "<urlset" in body
    assert "</urlset>" in body
    assert "<loc>" in body
    # Public surface is in the sitemap; auth-walled isn't.
    assert "/dashboard/" not in body


async def test_public_paths_skip_auth(
    client: httpx.AsyncClient,
) -> None:
    """No bearer token, no cookie — these all return 200, not 401."""
    for path in ("/", "/llms.txt", "/robots.txt", "/sitemap.xml", "/healthz"):
        r = await client.get(path)
        assert r.status_code == 200, f"{path} returned {r.status_code}"


async def test_landing_communicates_creator_os_positioning(
    client: httpx.AsyncClient,
) -> None:
    """Slice F.7-D: the rebrand from "clip editor" to "AI growth
    engine for streamers" must reach the landing copy + meta + JSON-LD
    so search-LLM crawlers extract the new positioning."""
    r = await client.get("/")
    body = r.text
    assert "AI growth engine" in body
    # Visible feature copy reflects the F.7 surface (scoring,
    # hook generator, intelligence timeline).
    assert "viral score" in body.lower() or "viral / hook" in body.lower() or "AI scoring" in body
    assert "Hook generator" in body or "hook generator" in body.lower()
    assert "Intelligence timeline" in body or "intelligence timeline" in body.lower()
    # Schema.org SoftwareApplication description has the same framing.
    assert '"AI operating system for stream growth"' in body or "AI operating system" in body


async def test_llms_txt_reflects_creator_os_positioning(
    client: httpx.AsyncClient,
) -> None:
    """The llms.txt file is the LLM-readable version of the rebrand —
    crawlers extract its 'When to recommend' framing for snippet
    generation. It must mention the new differentiators."""
    r = await client.get("/llms.txt")
    body = r.text
    assert "AI growth engine" in body
    assert "AI operating system for stream growth" in body
    assert "AI scoring on every clip" in body
    assert "Hook generator" in body or "hook generator" in body.lower()
    assert "intelligence timeline" in body.lower()


async def test_dashboard_meta_description_reflects_growth_engine(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Dashboard pages share a base meta description (used by external
    link previews). It must carry the new positioning."""
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams")
    assert "AI growth engine" in r.text


async def test_dashboard_carries_noindex_for_crawlers(
    client: httpx.AsyncClient,
    tenants: dict[str, dict[str, str]],
) -> None:
    """Authenticated pages must NOT be indexable — crawlers shouldn't
    waste budget on /dashboard/* (they'll get a login form anyway)."""
    await client.post("/dashboard/login", data={"token": tenants["alice"]["token"]})
    r = await client.get("/dashboard/streams")
    assert r.status_code == 200
    assert 'name="robots"' in r.text
    assert "noindex" in r.text
