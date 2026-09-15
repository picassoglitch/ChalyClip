"""Slice O.24 — request-level i18n for templates.

Detects the user's locale from the `Accept-Language` header and exposes
a `t(key)` function to Jinja templates that resolves to the localized
string. Spanish (`es`) + English (`en`), English fallback.

How a template uses it:
    {{ t('landing.hero.h1_line1') }}
    <html lang="{{ locale() }}">

VOICE NOTES (Spanish):
  - Mexican Spanish, TUTEO ("tú"). "sube", "transmite", "agenda", "di",
    "clipea", "tienes", "quieres", "puedes", "imagínate", "deja", "empieza".
    NOT Argentine voseo (no "subí / tenés / publicá / imaginate").
  - Direct-response, OpusClip-style. Short paragraphs. Declarative.
    Sell outcomes (more views, more time, less editing, less spend).
  - Numbers ratified by the operator: hero says "20+ clips ready to
    post", emotional checklist uses "17 clips generados" as a concrete
    realistic instance. These override the earlier "12+" guard rail.
  - Posting language: "Publica mientras duermes" is the headline copy on
    the feature card; the body sells automatic scheduling with a review /
    undo window. The undo window is the trust layer — keep it.

How to add a new key:
  1. Add to TRANSLATIONS['en'] AND TRANSLATIONS['es'].
  2. Reference in the template via `{{ t('your.key') }}`.
"""
from __future__ import annotations

from fastapi import Request
from fastapi.templating import Jinja2Templates

SUPPORTED_LOCALES: tuple[str, ...] = ("en", "es")
# Default to Mexican Spanish; English is served when the browser's
# Accept-Language asks for it (see detect_locale).
DEFAULT_LOCALE = "es"


def detect_locale(request: Request) -> str:
    """Return 'en' or 'es' based on the request's Accept-Language header."""
    raw = request.headers.get("accept-language", "")
    if not raw:
        return DEFAULT_LOCALE
    for tag in raw.split(","):
        primary = tag.split(";", 1)[0].strip().lower()
        primary = primary.split("-", 1)[0]
        if primary in SUPPORTED_LOCALES:
            return primary
    return DEFAULT_LOCALE


# ---- Translation registry -------------------------------------------------
TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        # ============================================================
        # LANDING PAGE — OpusClip-style outcome copy
        # ============================================================
        "landing.title_tag": (
            "ChalyClip — turn one stream into 20+ ready-to-post clips"
        ),

        # ---- Hero ---------------------------------------------------
        "landing.hero.eyebrow": "CHALYBCLIP",
        "landing.hero.h1_line1": "Stop paying editors.",
        "landing.hero.h1_line2": "Start publishing viral clips every day.",
        "landing.hero.sub_line1": (
            "Turn a 4-hour stream into 20+ ready-to-post clips in minutes."
        ),
        "landing.hero.sub_line2": (
            "ChalyClip finds the best moments, creates hooks, subtitles, "
            "captions, hashtags, crops for every platform, and schedules "
            "everything automatically."
        ),
        "landing.hero.tagline_a": "You stream.",
        "landing.hero.tagline_b": "ChalyClip handles the rest.",
        "landing.hero.cta_primary": "Upload my first stream",
        "landing.hero.cta_primary_micro": "Your first stream, free. No card.",
        "landing.hero.cta_secondary": "Watch demo",

        # ---- Trust bar ---------------------------------------------
        "landing.trust.t1": "TikTok Ready",
        "landing.trust.t2": "Instagram Reels",
        "landing.trust.t3": "YouTube Shorts",
        "landing.trust.t4": "Twitch VODs",
        "landing.trust.t5": "Kick Streams",

        # ---- Pain section ------------------------------------------
        "landing.pain.heading": "Editing clips is killing your growth.",
        "landing.pain.line1": "Every stream creates content.",
        "landing.pain.line2": "Most creators never get around to publishing it.",
        "landing.pain.line3": "Because editing takes:",
        "landing.pain.item1": "Hours of clipping",
        "landing.pain.item2": "Hours of captions",
        "landing.pain.item3": "Hours of thumbnails",
        "landing.pain.item4": "Hours of posting",
        "landing.pain.line4": "Or hundreds of dollars a month in editors.",
        "landing.pain.line5": "Meanwhile the stream is already dead.",

        # ---- "ChalyClip fixes that" ---------------------------------
        "landing.fix.heading": "ChalyClip fixes that.",
        "landing.fix.s1": "Upload stream.",
        "landing.fix.s2": "Get clips.",
        "landing.fix.s3": "Publish.",
        "landing.fix.s4": "Done.",

        # ---- Value Prop --------------------------------------------
        "landing.value.heading_line1": "One stream.",
        "landing.value.heading_line2": "Endless content.",
        "landing.value.intro": "A single stream becomes:",
        "landing.value.item1": "Viral clips",
        "landing.value.item2": "Hooks",
        "landing.value.item3": "Titles",
        "landing.value.item4": "Captions",
        "landing.value.item5": "Hashtags",
        "landing.value.item6": "Shorts",
        "landing.value.item7": "Reels",
        "landing.value.item8": "TikToks",
        "landing.value.without1": "Without touching Premiere.",
        "landing.value.without2": "Without hiring editors.",
        "landing.value.without3": "Without learning anything.",

        # ---- Money section -----------------------------------------
        "landing.math.heading": "What does one editor cost?",
        "landing.math.freelancer_label": "Freelance editor",
        "landing.math.freelancer_price": "$300 – $1,500 / month",
        "landing.math.agency_label": "Agency",
        "landing.math.agency_price": "$1,000 – $5,000 / month",
        "landing.math.us_label": "ChalyClip",
        # TODO(operator): replace $__ with the real launch price.
        "landing.math.us_price": "$__ / month",
        "landing.math.us_line1": "A fraction of the cost.",
        "landing.math.us_line2": "Works 24/7.",
        "landing.math.us_line3": "Never sleeps.",
        "landing.math.us_line4": "Never misses a stream.",
        "landing.math.us_line5": "Never takes vacation.",

        # ---- Emotional section -------------------------------------
        "landing.imagine.heading": "Imagine waking up to this.",
        "landing.imagine.line1": "You finish streaming at midnight.",
        "landing.imagine.line2": "You go to sleep.",
        "landing.imagine.line3": "When you wake up:",
        "landing.imagine.item1": "17 clips generated",
        "landing.imagine.item2": "Best moments identified",
        "landing.imagine.item3": "Hooks written",
        "landing.imagine.item4": "Captions created",
        "landing.imagine.item5": "Scheduled for TikTok",
        "landing.imagine.item6": "Scheduled for Reels",
        "landing.imagine.item7": "Scheduled for Shorts",
        "landing.imagine.tagline": "Your content machine never stopped working.",

        # ---- Results section ---------------------------------------
        "landing.results.heading": "More clips = More chances to go viral",
        "landing.results.line1": (
            "Creators who publish consistently grow faster."
        ),
        "landing.results.line2": "The problem isn't talent.",
        "landing.results.line3": "The problem is output.",
        "landing.results.line4": (
            "ChalyClip turns every stream into a content factory."
        ),

        # ---- Feature cards (rewritten) -----------------------------
        "landing.features.heading": "Everything you needed an editor for.",
        "landing.features.intro": "Automatic. Every night.",

        "landing.features.viral.title": (
            "Finds your viral moments automatically"
        ),
        "landing.features.viral.body": (
            "ChalyClip analyzes every second of your stream and identifies "
            "the clips most likely to perform. No manual reviewing."
        ),
        "landing.features.hook.title": "Never write another title again",
        "landing.features.hook.body": (
            "Get multiple viral title options for every clip. Choose one "
            "with one click."
        ),
        "landing.features.timeline.title": "Skip the boring parts",
        "landing.features.timeline.body": (
            "Jump directly to the funniest, loudest, most emotional and "
            "most engaging moments."
        ),
        "landing.features.voice.title": "Clip moments instantly",
        "landing.features.voice.body_html": (
            "Say: <code>clip that.</code> "
            "And ChalyClip remembers it automatically."
        ),
        "landing.features.brand.title": "Your content always looks on-brand",
        "landing.features.brand.body": (
            "Logos. Fonts. Colors. Captions. Everything applied automatically."
        ),
        "landing.features.publish.title": "Post while you sleep",
        "landing.features.publish.body": (
            "Schedule clips automatically across platforms. Review or undo "
            "anytime."
        ),

        # ---- Big statement -----------------------------------------
        "landing.statement_line1": "Stream more.",
        "landing.statement_line2": "Edit less.",
        "landing.statement_line3": "Grow faster.",

        # ---- Final CTA after statement -----------------------------
        "landing.final.heading": "Turn one stream into 20 pieces of content.",
        "landing.final.body": (
            "Finally grow your channel without hiring an editor or "
            "spending 4 hours a day cutting clips."
        ),
        "landing.final.cta": "Upload my first stream",
        "landing.final.cta_micro": "Free. No card.",

        # ---- FAQ ---------------------------------------------------
        "landing.faq.heading": "Frequently asked",
        "landing.faq.q1": (
            "How is ChalyClip different from a clip editor?"
        ),
        "landing.faq.a1": (
            "Editors give you a timeline and trim handles. ChalyClip is "
            "upstream: AI scores every clip on viral potential, generates "
            "the hook, applies the right brand kit, and schedules with an "
            "undo window. Your job shifts from 'find and cut clips' to "
            "'pick the AI's top 3 and let it ship'. The editor's still "
            "there when you need it — most mornings you won't."
        ),
        "landing.faq.q2": "Who is ChalyClip for?",
        "landing.faq.a2": (
            "Streamers who want clips ready by morning. Multi-host "
            "collectives that need per-streamer branding within one VOD. "
            "Agencies running clipping for several creators at once. Agents "
            "/ MCP clients that want to drive a clipping pipeline "
            "programmatically."
        ),
        "landing.faq.q3": "Do I need a GPU?",
        "landing.faq.a3": (
            "A consumer GPU (RTX 4060 or better recommended) handles "
            "Whisper + pyannote comfortably for ~4hr VODs. CPU-only works "
            "for short clips but is much slower."
        ),
        "landing.faq.q4": "Which LLM does ChalyClip use?",
        "landing.faq.a4": (
            "Anthropic Claude exclusively for the live surface. The router "
            "supports adding more providers later without code changes."
        ),
        "landing.faq.q5": "Can AI agents drive ChalyClip?",
        "landing.faq.a5_html": (
            "Yes. ChalyClip ships an MCP (Model Context Protocol) server "
            "for Claude Code, Cursor, and other LLM clients. Technical "
            "details on <a href=\"/agents\">/agents</a>."
        ),
        "landing.faq.q6": "What about retention and privacy?",
        "landing.faq.a6": (
            "Per-tenant retention windows (default 7 days for VODs, 90 "
            "for clips, 365 for transcripts) — all configurable. Audio and "
            "video stay on your machine for transcription; only LLM "
            "caption-generation calls out."
        ),
        "landing.faq.q7": "Is there a direct API?",
        "landing.faq.a7_html": (
            "Yes. <a href=\"/openapi.json\">OpenAPI spec</a> + "
            "<a href=\"/docs\">interactive Swagger UI</a>. Issue a token "
            "via <code>chalybclip tokens issue --tenant &lt;id&gt; --scope "
            "full</code> and pass as "
            "<code>Authorization: Bearer &lt;token&gt;</code>."
        ),

        # ---- Sticky nav (new — added when the reference port landed)
        "landing.nav.brand_prefix": "Chalyb",
        "landing.nav.brand_suffix": "Clip",
        "landing.nav.link_features": "Features",
        "landing.nav.link_solution": "Solution",
        "landing.nav.link_pricing": "Pricing",
        "landing.nav.link_agents": "Agencies / Agents",
        "landing.nav.link_api": "API",
        "landing.nav.signin": "Sign in",
        "landing.nav.cta": "Start free",

        # ---- Hero — paste-link box + alternate paths -------------
        "landing.hero.badge": (
            "AI clipping for streamers · #1 in LATAM"
        ),
        "landing.hero.paste_placeholder": (
            "Paste your VOD link — Kick, Twitch, YouTube…"
        ),
        "landing.hero.paste_cta": "Generate clips",
        "landing.hero.alt_or": "or",
        "landing.hero.alt_upload": "upload a file",
        "landing.hero.alt_micro": "Your first stream, free. No card.",
        "landing.hero.demo_label": "Watch demo:",
        "landing.hero.demo_gaming": "Gaming",
        "landing.hero.demo_chat": "Just Chatting",
        "landing.hero.demo_podcast": "Podcast",

        # ---- Hero — feature chips under the animation -----------
        "landing.chips.cut": "AI cutting",
        "landing.chips.subs": "Auto subtitles",
        "landing.chips.reframe": "9:16 reframe",
        "landing.chips.hooks": "Hooks & titles",
        "landing.chips.hashtags": "Hashtags",
        "landing.chips.schedule": "Multi-platform scheduling",

        # ---- ClipFanAnimation (source VOD card + 5 fanning
        #      vertical clips with a counting-up viral score) ------
        "landing.anim.source_label": "your_full_stream.mp4",
        "landing.anim.live_tag": "● LIVE · VOD",
        "landing.anim.score_label": "Score",
        "landing.anim.cap1": "\"You won't believe what happened…\"",
        "landing.anim.cap2": "The most viral moment of the stream",
        "landing.anim.cap3": "The reaction that broke it all",
        "landing.anim.cap4": "Clip you triggered with 'clip that'",
        "landing.anim.cap5": "Highlight with auto captions",
        "landing.anim.note": (
            "1 stream → 5 platforms → clips scored by viral "
            "potential, automatically"
        ),

        # ---- Footer (4-column layout) ----------------------------
        "landing.foot.tagline": (
            "ChalyClip · turn one stream into 20+ ready-to-post clips"
        ),
        "landing.foot.brand_blurb": (
            "Turn your streams into viral, ready-to-post clips. "
            "A Quantor · Chalyb product."
        ),
        "landing.foot.col_product": "Product",
        "landing.foot.col_developers": "Developers",
        "landing.foot.col_platforms": "Platforms",
        "landing.foot.link_features": "Features",
        "landing.foot.link_pricing": "Pricing",
        "landing.foot.link_get_started": "Start free",
        "landing.foot.link_openapi": "OpenAPI spec",
        "landing.foot.platform_tiktok": "TikTok",
        "landing.foot.platform_reels": "Instagram Reels",
        "landing.foot.platform_shorts": "YouTube Shorts",
        "landing.foot.platform_kick": "Kick · Twitch VODs",
        "landing.foot.copyright": (
            "© 2026 ChalyClip · Quantor. All rights reserved."
        ),
        "landing.foot.made": "Built in Chalyb World ◇",
        "landing.foot.link_agents": "For agents / agencies",
        "landing.foot.link_llms": "llms.txt",
        "landing.foot.link_docs": "API docs",
        "landing.foot.link_signin": "Sign in",

        # ============================================================
        # /agents PAGE — the technical second door (unchanged from
        # previous version — already lives on /agents with deep tech).
        # ============================================================
        "agents.title_tag": (
            "ChalyClip for agencies, integrators and AI agents"
        ),
        "agents.hero.eyebrow": "FOR AGENCIES · AGENTS · INTEGRATORS",
        "agents.hero.h1": (
            "Built for the era where the operator is half-human, half-agent."
        ),
        "agents.hero.sub": (
            "Multimodal signal fan-in, per-speaker diarization, local GPU "
            "transcription, MCP-native control surface, fully-typed REST API. "
            "Drive ChalyClip from a browser, Claude Code, Cursor, or any "
            "MCP-aware client."
        ),
        "agents.hero.cta_docs": "OpenAPI docs",
        "agents.hero.cta_llms": "llms.txt",
        "agents.hero.cta_back": "← Back to the landing",
        "agents.signals.title": "Multimodal signal fan-in",
        "agents.signals.body": (
            "Voice-marker triggers, chat heat, audio peaks, scene cuts, "
            "face presence and motion all feed a single candidate-window "
            "scorer. Each candidate carries viral score, hook strength, "
            "caption readability and dead-air risk."
        ),
        "agents.diarization.title": "Per-speaker diarization",
        "agents.diarization.body": (
            "pyannote-audio 3.1 labels every speech segment; cosine-"
            "similarity embeddings resolve persistent identities across "
            "VODs. Multi-host streams route each clip to the right host's "
            "brand kit — without manual tagging."
        ),
        "agents.gpu.title": "Local GPU transcription",
        "agents.gpu.body": (
            "faster-whisper runs on the operator's own GPU. Stream audio "
            "never leaves the machine for transcription. The only LLM "
            "round-trip is caption + hook generation through Anthropic "
            "Claude."
        ),
        "agents.mcp.title": "MCP-native control surface",
        "agents.mcp.body_html": (
            "Every operator action is also an MCP tool. Run "
            "<code>chalybclip mcp serve --token &lt;api-token&gt;</code> to "
            "expose stdio MCP. Claude Code, Cursor, and any other MCP "
            "client see tools for listing streams, kicking off pipelines, "
            "fetching clips, managing brand kits, inspecting LLM spend."
        ),
        "agents.api.title": "Fully-typed REST + OpenAPI",
        "agents.api.body_html": (
            "Auto-generated OpenAPI 3.1 spec at <code>/openapi.json</code>, "
            "interactive Swagger at <code>/docs</code>. Pydantic-validated "
            "request and response bodies; no string parsing on either side."
        ),
        "agents.persona.title": (
            "Half-human, half-agent operator"
        ),
        "agents.persona.body": (
            "ChalyClip ships assuming the operator is a hybrid: an agent "
            "that drafts the batch overnight, a human that approves it in "
            "30 seconds with morning coffee. The undo window is the trust "
            "boundary between those two halves."
        ),
        "agents.cta.heading": "Start integrating",
        "agents.cta.body": (
            "Read the OpenAPI spec, drop the llms.txt into your LLM "
            "client's context, or run the MCP server locally. The tenant "
            "token is the only credential you need."
        ),

        # ============================================================
        # NAV (dashboard chrome — base.html)
        # ============================================================
        "nav.clips": "Streams",
        "nav.inbox": "Clips",
        "nav.personas": "Personas",
        "nav.brand_kits": "Brand kits",
        "nav.publish": "Publish",
        "nav.sources": "Channels",
        "nav.llm_spend": "LLM spend",
        "nav.llm_settings": "LLM settings",
        "nav.site_settings": "Site / Landing",
        "nav.docs_agents": "Docs / Agents",
        "nav.logout": "Logout",
        "nav.new_clip": "New clip",
        # ---- New clip — ingest hub (single hero card + mode selector) ----
        "newClip.modesHint": "or send it another way",
        "newClip.modes.url.label": "Paste URL",
        "newClip.modes.upload.label": "Upload",
        "newClip.modes.live.label": "Push live",
        "newClip.modes.drive.label": "Drive folder",
        "newClip.url.title": "Paste a URL",
        "newClip.url.tag": "Fastest",
        "newClip.url.help": "Kick, Twitch, or YouTube VOD. We download it for you.",
        "newClip.url.fieldLabel": "VOD URL",
        "newClip.url.placeholder": "https://kick.com/aldovillanueva/videos/…",
        "newClip.url.supportedPre": "Supported:",
        "newClip.url.start": "Start",
        "newClip.url.cookiesPre": "Kick + age-gated YouTube need a logged-in cookies.txt on the volume — set",
        "newClip.url.cookiesMid": "to point at it.",
        "newClip.upload.title": "Upload a video",
        "newClip.upload.tag": "No URL needed",
        "newClip.upload.help": "Already-recorded VOD or phone video.",
        "newClip.upload.fieldLabel": "Video file",
        "newClip.upload.dropTitle": "Drop a file or click to browse",
        "newClip.upload.dropHint": "Drag your VOD here, or select it from your device",
        "newClip.upload.cta": "Upload & process",
        "newClip.upload.largeFiles": "Large files (> 500 MB) — keep the tab open until the upload finishes.",
        "newClip.live.title": "Push live (RTMP)",
        "newClip.live.tag": "OBS / Streamlabs",
        "newClip.live.help": "Paste these into OBS. We record the live push and run the pipeline when you stop the broadcast.",
        "newClip.live.serverLabel": "Server",
        "newClip.live.keyLabel": "Stream key",
        "newClip.live.reveal": "Reveal",
        "newClip.live.hide": "Hide",
        "newClip.live.copy": "Copy",
        "newClip.live.copied": "Copied",
        "newClip.live.openDashboard": "Open live dashboard",
        "newClip.live.note": "We start recording the moment OBS connects. Stop the broadcast to trigger the pipeline automatically.",
        "newClip.live.notConfigured": "Not configured on this deployment. Set CHALYBCLIP_LIVE_RTMP_BASE_URL in env to enable.",
        "newClip.live.unavailable": "Not available",
        "newClip.drive.title": "Watch a Drive folder",
        "newClip.drive.tag": "Auto-ingest",
        "newClip.drive.help": "Drop new VODs into a Google Drive folder; we pull each one automatically. Good for team workflows.",
        "newClip.drive.watchedLabel": "Watched folder",
        "newClip.drive.connectTitle": "Connect a Drive folder",
        "newClip.drive.connectHint": "Coming soon — Drive folder auto-ingest isn't available yet.",
        "newClip.drive.manage": "Manage Drive watches",
        "newClip.drive.unavailable": "Coming soon",
        "newClip.drive.note": "In the meantime, use Paste URL or Upload to bring in a VOD.",

        # ============================================================
        # SITE SETTINGS (admin — operator-editable landing values)
        # ============================================================
        "settings.site.title": "Site settings",
        "settings.site.heading": "Site / landing",
        "settings.site.intro": (
            "Operator-editable values shown on the public landing page. "
            "Changes go live immediately — no redeploy."
        ),
        "settings.site.section_price": "Landing price",
        "settings.site.price_help": (
            "shown in the 'what does an editor cost' comparison"
        ),
        "settings.site.price_es_label": "Price (Spanish landing)",
        "settings.site.price_en_label": "Price (English landing)",
        "settings.site.placeholder_es": "e.g. US$ 29 / mes",
        "settings.site.placeholder_en": "e.g. $29 / month",
        "settings.site.preview_note": (
            "Type the full string exactly as it should appear, including "
            "currency and period (e.g. '$29 / month'). Leave a field blank "
            "to fall back to the other language, or to the default "
            "placeholder if both are empty."
        ),
        "settings.site.save": "Save",
        "settings.site.saved": "Saved. The landing is updated.",
        "settings.site.view_landing": "View landing",

        # ============================================================
        # SSO failure page
        # ============================================================
        "sso.fail.title": "Invalid session",
        "sso.fail.body": (
            "The link you used to enter is invalid or expired. Go back to "
            "the Chalyb dashboard and open ChalyClip again."
        ),
        "sso.fail.cta": "Go to Chalyb",
    },

    # ================================================================
    # SPANISH — voseo platense/latino, OpusClip rhythm
    # ================================================================
    "es": {
        "landing.title_tag": (
            "ChalyClip — convierte un stream en 20+ clips listos para publicar"
        ),

        # ---- Hero ---------------------------------------------------
        "landing.hero.eyebrow": "CHALYBCLIP",
        "landing.hero.h1_line1": "Deja de pagar editores.",
        "landing.hero.h1_line2": "Empieza a publicar clips virales todos los días.",
        "landing.hero.sub_line1": (
            "Convierte un stream de 4 horas en 20+ clips listos para "
            "publicar, en minutos."
        ),
        "landing.hero.sub_line2": (
            "ChalyClip encuentra los mejores momentos, crea hooks, "
            "subtítulos, captions, hashtags, recorta para cada plataforma, "
            "y agenda todo automáticamente."
        ),
        "landing.hero.tagline_a": "Tú transmites.",
        "landing.hero.tagline_b": "ChalyClip hace el resto.",
        "landing.hero.cta_primary": "Subir mi primer stream",
        "landing.hero.cta_primary_micro": "Tu primer stream, gratis. Sin tarjeta.",
        "landing.hero.cta_secondary": "Ver demo",

        # ---- Trust bar ---------------------------------------------
        "landing.trust.t1": "TikTok",
        "landing.trust.t2": "Instagram Reels",
        "landing.trust.t3": "YouTube Shorts",
        "landing.trust.t4": "Twitch VODs",
        "landing.trust.t5": "Kick Streams",

        # ---- Pain section ------------------------------------------
        "landing.pain.heading": "Editar clips te está matando el crecimiento.",
        "landing.pain.line1": "Cada stream genera contenido.",
        "landing.pain.line2": "Casi ningún creador llega a publicarlo.",
        "landing.pain.line3": "Porque editar lleva:",
        "landing.pain.item1": "Horas de clipping",
        "landing.pain.item2": "Horas de captions",
        "landing.pain.item3": "Horas de miniaturas",
        "landing.pain.item4": "Horas de publicar",
        "landing.pain.line4": "O cientos de dólares al mes en editores.",
        "landing.pain.line5": "Mientras tanto el stream ya está muerto.",

        # ---- "ChalyClip fixes that" ---------------------------------
        "landing.fix.heading": "ChalyClip lo arregla.",
        "landing.fix.s1": "Sube el stream.",
        "landing.fix.s2": "Recibe los clips.",
        "landing.fix.s3": "Publica.",
        "landing.fix.s4": "Listo.",

        # ---- Value Prop --------------------------------------------
        "landing.value.heading_line1": "Un stream.",
        "landing.value.heading_line2": "Contenido infinito.",
        "landing.value.intro": "Un solo stream se convierte en:",
        "landing.value.item1": "Clips virales",
        "landing.value.item2": "Hooks",
        "landing.value.item3": "Títulos",
        "landing.value.item4": "Captions",
        "landing.value.item5": "Hashtags",
        "landing.value.item6": "Shorts",
        "landing.value.item7": "Reels",
        "landing.value.item8": "TikToks",
        "landing.value.without1": "Sin tocar Premiere.",
        "landing.value.without2": "Sin contratar editores.",
        "landing.value.without3": "Sin aprender nada.",

        # ---- Money section -----------------------------------------
        "landing.math.heading": "¿Cuánto cuesta un editor?",
        "landing.math.freelancer_label": "Editor freelance",
        "landing.math.freelancer_price": "$300 – $1.500 / mes",
        "landing.math.agency_label": "Agencia",
        "landing.math.agency_price": "$1.000 – $5.000 / mes",
        "landing.math.us_label": "ChalyClip",
        # TODO(operator): poner el precio real de launch.
        "landing.math.us_price": "$__ / mes",
        "landing.math.us_line1": "Una fracción del costo.",
        "landing.math.us_line2": "Trabaja 24/7.",
        "landing.math.us_line3": "Nunca duerme.",
        "landing.math.us_line4": "Nunca se pierde un stream.",
        "landing.math.us_line5": "Nunca se va de vacaciones.",

        # ---- Emotional section -------------------------------------
        "landing.imagine.heading": "Imagínate despertar con esto.",
        "landing.imagine.line1": "Terminas de transmitir a medianoche.",
        "landing.imagine.line2": "Te vas a dormir.",
        "landing.imagine.line3": "Cuando despiertas:",
        "landing.imagine.item1": "17 clips generados",
        "landing.imagine.item2": "Mejores momentos identificados",
        "landing.imagine.item3": "Hooks escritos",
        "landing.imagine.item4": "Captions creados",
        "landing.imagine.item5": "Agendados para TikTok",
        "landing.imagine.item6": "Agendados para Reels",
        "landing.imagine.item7": "Agendados para Shorts",
        "landing.imagine.tagline": (
            "Tu máquina de contenido nunca paró de trabajar."
        ),

        # ---- Results section ---------------------------------------
        "landing.results.heading": (
            "Más clips = más chances de hacer viral"
        ),
        "landing.results.line1": (
            "Los creadores que publican constante crecen más rápido."
        ),
        "landing.results.line2": "El problema no es el talento.",
        "landing.results.line3": "El problema es el output.",
        "landing.results.line4": (
            "ChalyClip convierte cada stream en una fábrica de contenido."
        ),

        # ---- Feature cards (rewritten) -----------------------------
        "landing.features.heading": (
            "Todo lo que necesitabas un editor para hacer."
        ),
        "landing.features.intro": "Automático. Cada noche.",

        "landing.features.viral.title": (
            "Encuentra tus momentos virales solo"
        ),
        "landing.features.viral.body": (
            "ChalyClip analiza cada segundo de tu stream e identifica los "
            "clips con más chances de pegar. Sin revisar nada a mano."
        ),
        "landing.features.hook.title": "No vuelves a escribir un título",
        "landing.features.hook.body": (
            "Recibe varias opciones de título viral para cada clip. "
            "Elige una con un clic."
        ),
        "landing.features.timeline.title": "Sáltate las partes aburridas",
        "landing.features.timeline.body": (
            "Ve directo a los momentos más divertidos, ruidosos, "
            "emocionantes y que más enganchan."
        ),
        "landing.features.voice.title": "Clipea momentos al instante",
        "landing.features.voice.body_html": (
            "Di: <code>clipea esto.</code> "
            "Y ChalyClip lo recuerda automáticamente."
        ),
        "landing.features.brand.title": (
            "Tu contenido siempre se ve on-brand"
        ),
        "landing.features.brand.body": (
            "Logos. Fuentes. Colores. Captions. Todo aplicado automáticamente."
        ),
        "landing.features.publish.title": "Publica mientras duermes",
        "landing.features.publish.body": (
            "Programa clips automáticamente en todas las plataformas. "
            "Revisa o deshaz cuando quieras."
        ),

        # ---- Big statement -----------------------------------------
        "landing.statement_line1": "Transmite más.",
        "landing.statement_line2": "Edita menos.",
        "landing.statement_line3": "Crece más rápido.",

        # ---- Final CTA --------------------------------------------
        "landing.final.heading": (
            "Convierte un stream en 20 piezas de contenido."
        ),
        "landing.final.body": (
            "Por fin haz crecer tu canal sin contratar un editor y "
            "sin pasarte 4 horas al día cortando clips."
        ),
        "landing.final.cta": "Subir mi primer stream",
        "landing.final.cta_micro": "Gratis. Sin tarjeta.",

        # ---- FAQ ---------------------------------------------------
        "landing.faq.heading": "Preguntas frecuentes",
        "landing.faq.q1": "¿En qué se diferencia ChalyClip de un editor de clips?",
        "landing.faq.a1": (
            "Los editores te dan una línea de tiempo con manijas para "
            "recortar. ChalyClip va un paso antes: la IA puntúa cada clip "
            "por potencial viral, genera el hook, aplica el brand kit "
            "correcto y lo programa con ventana para deshacer. Tu trabajo "
            "cambia de \"encontrar y cortar\" a \"elegir el top 3 de la IA "
            "y dejar que se publique\". El editor sigue ahí — pero casi "
            "ninguna mañana lo vas a abrir."
        ),
        "landing.faq.q2": "¿Para quién es ChalyClip?",
        "landing.faq.a2": (
            "Streamers que quieren los clips listos para la mañana. "
            "Colectivos multi-host que necesitan branding por streamer "
            "dentro de un mismo VOD. Agencias que hacen clips para varios "
            "creadores en paralelo. Agentes / clientes MCP que quieren "
            "manejar el pipeline de forma programática."
        ),
        "landing.faq.q3": "¿Necesito una GPU?",
        "landing.faq.a3": (
            "Una GPU de consumidor (RTX 4060 o mejor, recomendada) maneja "
            "Whisper + pyannote cómodamente para VODs de ~4 horas. Solo "
            "con CPU funciona para clips cortos, pero bastante más lento."
        ),
        "landing.faq.q4": "¿Qué LLM usa ChalyClip?",
        "landing.faq.a4": (
            "Anthropic Claude en exclusiva para todo lo que ves en vivo. "
            "El router permite sumar más proveedores sin tocar el código."
        ),
        "landing.faq.q5": "¿Los agentes de IA pueden manejar ChalyClip?",
        "landing.faq.a5_html": (
            "Sí. ChalyClip incluye un servidor MCP para Claude Code, Cursor "
            "y otros clientes LLM. Detalles técnicos en "
            "<a href=\"/agents\">/agents</a>."
        ),
        "landing.faq.q6": "¿Qué pasa con la retención y la privacidad?",
        "landing.faq.a6": (
            "Ventanas de retención por tenant (por defecto 7 días para "
            "VODs, 90 para clips, 365 para transcripciones) — todo "
            "configurable. El audio y el video se quedan en tu máquina "
            "para transcribir; solo la generación de captions con LLM "
            "sale hacia afuera."
        ),
        "landing.faq.q7": "¿Hay una API directa que pueda llamar?",
        "landing.faq.a7_html": (
            "Sí. <a href=\"/openapi.json\">Spec OpenAPI</a> + "
            "<a href=\"/docs\">Swagger UI interactivo</a>. Generas un "
            "token con <code>chalybclip tokens issue --tenant &lt;id&gt; "
            "--scope full</code> y lo pasas como "
            "<code>Authorization: Bearer &lt;token&gt;</code>."
        ),

        # ---- Sticky nav (new — added when the reference port landed)
        "landing.nav.brand_prefix": "Chalyb",
        "landing.nav.brand_suffix": "Clip",
        "landing.nav.link_features": "Características",
        "landing.nav.link_solution": "Solución",
        "landing.nav.link_pricing": "Precios",
        "landing.nav.link_agents": "Agencias / Agentes",
        "landing.nav.link_api": "API",
        "landing.nav.signin": "Iniciar sesión",
        "landing.nav.cta": "Empezar gratis",

        # ---- Hero — paste-link box + alternate paths -------------
        "landing.hero.badge": (
            "Clipping con IA para streamers · #1 en LATAM"
        ),
        "landing.hero.paste_placeholder": (
            "Pega el link de tu VOD — Kick, Twitch, YouTube…"
        ),
        "landing.hero.paste_cta": "Generar clips",
        "landing.hero.alt_or": "o",
        "landing.hero.alt_upload": "subir un archivo",
        "landing.hero.alt_micro": "Tu primer stream gratis. Sin tarjeta.",
        "landing.hero.demo_label": "Ver demo:",
        "landing.hero.demo_gaming": "Gaming",
        "landing.hero.demo_chat": "Just Chatting",
        "landing.hero.demo_podcast": "Podcast",

        # ---- Hero — feature chips under the animation -----------
        "landing.chips.cut": "Recorte con IA",
        "landing.chips.subs": "Subtítulos automáticos",
        "landing.chips.reframe": "Reencuadre 9:16",
        "landing.chips.hooks": "Hooks y títulos",
        "landing.chips.hashtags": "Hashtags",
        "landing.chips.schedule": "Programación multiplataforma",

        # ---- ClipFanAnimation (tarjeta VOD origen + 5 clips
        #      verticales en abanico con un score viral que cuenta
        #      hacia su valor al entrar en pantalla) ----------------
        "landing.anim.source_label": "tu_stream_completo.mp4",
        "landing.anim.live_tag": "● EN VIVO · VOD",
        "landing.anim.score_label": "Score",
        "landing.anim.cap1": "\"No vas a creer lo que pasó…\"",
        "landing.anim.cap2": "El momento más viral del directo",
        "landing.anim.cap3": "La reacción que lo rompió todo",
        "landing.anim.cap4": "Clip que pediste con \"clipea eso\"",
        "landing.anim.cap5": "Highlight con subtítulos automáticos",
        "landing.anim.note": (
            "1 stream → 5 plataformas → clips puntuados por "
            "potencial viral, automáticamente"
        ),

        # ---- Footer (4-column layout) ----------------------------
        "landing.foot.tagline": (
            "ChalyClip · convierte un stream en 20+ clips listos para publicar"
        ),
        "landing.foot.brand_blurb": (
            "Convierte tus streams en clips virales listos para "
            "publicar. Un producto de Quantor · Chalyb."
        ),
        "landing.foot.col_product": "Producto",
        "landing.foot.col_developers": "Desarrolladores",
        "landing.foot.col_platforms": "Plataformas",
        "landing.foot.link_features": "Características",
        "landing.foot.link_pricing": "Precios",
        "landing.foot.link_get_started": "Empezar gratis",
        "landing.foot.link_openapi": "Spec OpenAPI",
        "landing.foot.platform_tiktok": "TikTok",
        "landing.foot.platform_reels": "Instagram Reels",
        "landing.foot.platform_shorts": "YouTube Shorts",
        "landing.foot.platform_kick": "Kick · Twitch VODs",
        "landing.foot.copyright": (
            "© 2026 ChalyClip · Quantor. Todos los derechos reservados."
        ),
        "landing.foot.made": "Construido en Chalyb World ◇",
        "landing.foot.link_agents": "Para agentes / agencias",
        "landing.foot.link_llms": "llms.txt",
        "landing.foot.link_docs": "Docs API",
        "landing.foot.link_signin": "Ingresar",

        # ============================================================
        # /agents PAGE — second door technical (unchanged)
        # ============================================================
        "agents.title_tag": (
            "ChalyClip para agencias, integradores y agentes de IA"
        ),
        "agents.hero.eyebrow": "PARA AGENCIAS · AGENTES · INTEGRADORES",
        "agents.hero.h1": (
            "Construido para la era donde el operador es mitad humano, "
            "mitad agente."
        ),
        "agents.hero.sub": (
            "Fan-in de señales multimodales, diarización por speaker, "
            "transcripción GPU local, surface nativa de MCP, REST API "
            "tipada de punta a punta. Maneja ChalyClip desde un navegador, "
            "desde Claude Code, Cursor, o cualquier cliente compatible con MCP."
        ),
        "agents.hero.cta_docs": "Docs OpenAPI",
        "agents.hero.cta_llms": "llms.txt",
        "agents.hero.cta_back": "← Volver al landing",
        "agents.signals.title": "Fan-in de señales multimodales",
        "agents.signals.body": (
            "Marcadores de voz, calor del chat, picos de audio, cortes de "
            "escena, presencia de cara y movimiento alimentan un solo "
            "scorer de ventanas candidatas. Cada candidata trae puntaje "
            "viral, fuerza del hook, legibilidad del caption y riesgo de "
            "silencio en cámara."
        ),
        "agents.diarization.title": "Diarización por speaker",
        "agents.diarization.body": (
            "pyannote-audio 3.1 etiqueta cada segmento de habla; los "
            "embeddings por similitud de coseno resuelven identidades "
            "persistentes entre VODs. Los streams multi-host mandan cada "
            "clip al brand kit del host correcto — sin etiquetar a mano."
        ),
        "agents.gpu.title": "Transcripción GPU local",
        "agents.gpu.body": (
            "faster-whisper corre en la GPU del operador. El audio del "
            "stream nunca sale de la máquina para transcribir. La única "
            "ida y vuelta al LLM es la generación de captions + hooks vía "
            "Anthropic Claude."
        ),
        "agents.mcp.title": "Surface de control nativa de MCP",
        "agents.mcp.body_html": (
            "Cada acción del operador es también una herramienta MCP. Corre "
            "<code>chalybclip mcp serve --token &lt;api-token&gt;</code> "
            "para exponer MCP por stdio. Claude Code, Cursor y cualquier "
            "cliente MCP ven herramientas para listar streams, arrancar "
            "pipelines, traer clips, manejar brand kits e inspeccionar "
            "el gasto en LLM."
        ),
        "agents.api.title": "REST tipada de punta a punta + OpenAPI",
        "agents.api.body_html": (
            "Spec OpenAPI 3.1 autogenerado en <code>/openapi.json</code>, "
            "Swagger interactivo en <code>/docs</code>. Request y response "
            "validados con Pydantic; cero parseo de strings en ambos "
            "lados."
        ),
        "agents.persona.title": (
            "Operador: mitad humano, mitad agente"
        ),
        "agents.persona.body": (
            "ChalyClip se diseñó asumiendo que el operador es híbrido: un "
            "agente que arma el batch durante la noche, un humano que "
            "lo aprueba en 30 segundos con el café de la mañana. La "
            "ventana para deshacer es el límite de confianza entre las "
            "dos mitades."
        ),
        "agents.cta.heading": "Empieza a integrar",
        "agents.cta.body": (
            "Lee el spec OpenAPI, suelta el llms.txt en el contexto de "
            "tu cliente LLM, o corre el servidor MCP local. El token de "
            "tenant es la única credencial que necesitas."
        ),

        # ============================================================
        # NAV
        # ============================================================
        "nav.clips": "Streams",
        "nav.inbox": "Clips",
        "nav.personas": "Personas",
        "nav.brand_kits": "Brand kits",
        "nav.publish": "Publicar",
        "nav.sources": "Canales",
        "nav.llm_spend": "Gasto LLM",
        "nav.llm_settings": "Ajustes LLM",
        "nav.site_settings": "Sitio / Landing",
        "nav.docs_agents": "Docs / Agentes",
        "nav.logout": "Salir",
        "nav.new_clip": "Nuevo clip",
        # ---- Nuevo clip — hub de ingesta (una sola card + selector) ----
        "newClip.modesHint": "o mándalo de otra forma",
        "newClip.modes.url.label": "Pegar URL",
        "newClip.modes.upload.label": "Subir",
        "newClip.modes.live.label": "En vivo",
        "newClip.modes.drive.label": "Carpeta Drive",
        "newClip.url.title": "Pega una URL",
        "newClip.url.tag": "Lo más rápido",
        "newClip.url.help": "VOD de Kick, Twitch o YouTube. Lo descargamos por ti.",
        "newClip.url.fieldLabel": "URL del VOD",
        "newClip.url.placeholder": "https://kick.com/aldovillanueva/videos/…",
        "newClip.url.supportedPre": "Soportados:",
        "newClip.url.start": "Empezar",
        "newClip.url.cookiesPre": "Kick y YouTube con restricción de edad necesitan un cookies.txt con sesión iniciada en el volumen — define",
        "newClip.url.cookiesMid": "para apuntarlo.",
        "newClip.upload.title": "Sube un video",
        "newClip.upload.tag": "Sin URL",
        "newClip.upload.help": "VOD ya grabado o video del celular.",
        "newClip.upload.fieldLabel": "Archivo de video",
        "newClip.upload.dropTitle": "Suelta un archivo o haz clic para buscar",
        "newClip.upload.dropHint": "Arrastra tu VOD aquí, o selecciónalo desde tu dispositivo",
        "newClip.upload.cta": "Subir y procesar",
        "newClip.upload.largeFiles": "Archivos grandes (> 500 MB) — deja la pestaña abierta hasta que termine la subida.",
        "newClip.live.title": "Transmite en vivo (RTMP)",
        "newClip.live.tag": "OBS / Streamlabs",
        "newClip.live.help": "Pega esto en OBS. Grabamos la transmisión y corremos el pipeline cuando la cortas.",
        "newClip.live.serverLabel": "Servidor",
        "newClip.live.keyLabel": "Clave de transmisión",
        "newClip.live.reveal": "Mostrar",
        "newClip.live.hide": "Ocultar",
        "newClip.live.copy": "Copiar",
        "newClip.live.copied": "Copiado",
        "newClip.live.openDashboard": "Abrir panel en vivo",
        "newClip.live.note": "Empezamos a grabar apenas OBS se conecta. Corta la transmisión para disparar el pipeline automáticamente.",
        "newClip.live.notConfigured": "No está configurado en este despliegue. Define CHALYBCLIP_LIVE_RTMP_BASE_URL en el env para habilitarlo.",
        "newClip.live.unavailable": "No disponible",
        "newClip.drive.title": "Vigila una carpeta de Drive",
        "newClip.drive.tag": "Auto-ingesta",
        "newClip.drive.help": "Suelta nuevos VODs en una carpeta de Google Drive; los traemos uno por uno automáticamente. Ideal para equipos.",
        "newClip.drive.watchedLabel": "Carpeta vigilada",
        "newClip.drive.connectTitle": "Conecta una carpeta de Drive",
        "newClip.drive.connectHint": "Próximamente — la auto-ingesta de carpetas de Drive aún no está disponible.",
        "newClip.drive.manage": "Gestionar carpetas de Drive",
        "newClip.drive.unavailable": "Próximamente",
        "newClip.drive.note": "Mientras tanto, usa Pegar URL o Subir para traer un VOD.",

        # ============================================================
        # SITE SETTINGS (admin — valores del landing editables por el operador)
        # ============================================================
        "settings.site.title": "Ajustes del sitio",
        "settings.site.heading": "Sitio / landing",
        "settings.site.intro": (
            "Valores editables por el operador que se muestran en el landing "
            "público. Los cambios salen al instante — sin redeploy."
        ),
        "settings.site.section_price": "Precio del landing",
        "settings.site.price_help": (
            "se muestra en la comparación '¿cuánto cuesta un editor?'"
        ),
        "settings.site.price_es_label": "Precio (landing en español)",
        "settings.site.price_en_label": "Precio (landing en inglés)",
        "settings.site.placeholder_es": "ej. US$ 29 / mes",
        "settings.site.placeholder_en": "ej. $29 / month",
        "settings.site.preview_note": (
            "Escribe el texto completo tal cual quieres que aparezca, "
            "incluyendo la moneda y el período (ej. 'US$ 29 / mes'). Deja un "
            "campo vacío para que use el otro idioma, o el placeholder por "
            "defecto si los dos están vacíos."
        ),
        "settings.site.save": "Guardar",
        "settings.site.saved": "Guardado. El landing quedó actualizado.",
        "settings.site.view_landing": "Ver landing",

        # ============================================================
        # SSO failure page
        # ============================================================
        "sso.fail.title": "Sesión inválida",
        "sso.fail.body": (
            "El enlace que usaste para entrar no es válido o ya expiró. "
            "Vuelve al dashboard de Chalyb y abre ChalyClip de nuevo."
        ),
        "sso.fail.cta": "Ir a Chalyb",
    },
}


def t(key: str, locale: str = DEFAULT_LOCALE) -> str:
    """Resolve a translation key for the given locale, falling back to
    English if the key is missing, falling back to the key itself if
    English also doesn't have it (which surfaces the gap loudly).
    """
    table = TRANSLATIONS.get(locale) or TRANSLATIONS[DEFAULT_LOCALE]
    if key in table:
        return table[key]
    en_table = TRANSLATIONS[DEFAULT_LOCALE]
    if key in en_table:
        return en_table[key]
    return key


def install_globals(templates: Jinja2Templates) -> None:
    """Wire `t()` and `locale()` as Jinja globals on a Jinja2Templates
    instance. Call once per templates instance at app boot.
    """
    import jinja2

    @jinja2.pass_context
    def _t_for_request(ctx, key: str) -> str:
        request = ctx.get("request")
        locale = (
            getattr(request.state, "locale", DEFAULT_LOCALE)
            if request else DEFAULT_LOCALE
        )
        return t(key, locale)

    @jinja2.pass_context
    def _locale_for_request(ctx) -> str:
        request = ctx.get("request")
        return (
            getattr(request.state, "locale", DEFAULT_LOCALE)
            if request else DEFAULT_LOCALE
        )

    templates.env.globals["t"] = _t_for_request
    templates.env.globals["locale"] = _locale_for_request


__all__ = [
    "SUPPORTED_LOCALES",
    "DEFAULT_LOCALE",
    "TRANSLATIONS",
    "detect_locale",
    "t",
    "install_globals",
]
