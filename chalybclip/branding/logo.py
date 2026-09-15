"""AI logo generation — voice-markers spec §3.3.a.

Pipeline:
    Anthropic call → raw SVG → sanitize → save SVG + optional PNG raster

Sanitization is a hard allowlist on SVG elements + attributes. Anything
not on the list gets stripped; the resulting SVG is safe to inline-embed
in the dashboard without XSS risk.

Rasterization is optional. When cairosvg + a Cairo runtime are available,
we write a 1024x1024 PNG alongside the SVG so the ffmpeg overlay step
(slice D's logo burn-in) can use it. When they're not, the SVG-only
output is still useful — the dashboard renders SVGs natively, and the
PNG step is rerunnable later via a regenerate flow.

Cost: one Anthropic 'logo_generation' call per request. The router's
existing budget governor + cost log handles billing. Tenants can
generate up to N logos before hitting their daily LLM budget cap.
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

import structlog
from pydantic import BaseModel, ConfigDict, Field

from chalybclip.llm import LLMRouter

_log = structlog.get_logger(__name__)

# Register the SVG namespace as the default (empty-prefix) so the
# serializer emits `<svg ...>` rather than `<ns0:svg ...>`. Module-level
# is the right place: ET's namespace registry is global and we want every
# sanitize_svg() call to produce identical output.
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


# ---- schema returned by the LLM ----


class LogoSVG(BaseModel):
    """Structured-output schema for the logo prompt.

    The LLM returns valid SVG code (just the root <svg> ... </svg>) plus a
    one-line style hint we surface in the dashboard so the operator can
    see why this particular variant was generated.
    """

    model_config = ConfigDict(extra="forbid")
    svg: str = Field(min_length=20)
    style_hint: str = Field(min_length=1, max_length=200)


# ---- SVG sanitization ----


# Strict allowlist. Anything not here gets dropped on the floor. Keep it
# minimal — only elements we expect logo-style SVGs to use.
_ALLOWED_TAGS = frozenset(
    {
        "svg",
        "g",
        "defs",
        "title",
        "desc",
        "path",
        "rect",
        "circle",
        "ellipse",
        "line",
        "polyline",
        "polygon",
        "text",
        "tspan",
        "linearGradient",
        "radialGradient",
        "stop",
        "clipPath",
        "mask",
        "use",  # use is allowed but only with #-prefix refs (see below)
    }
)
_ALLOWED_ATTRS = frozenset(
    {
        "id",
        "class",
        "viewBox",
        "xmlns",
        "xmlns:xlink",
        "width",
        "height",
        "x",
        "y",
        "x1",
        "y1",
        "x2",
        "y2",
        "cx",
        "cy",
        "r",
        "rx",
        "ry",
        "d",
        "points",
        "fill",
        "fill-opacity",
        "fill-rule",
        "stroke",
        "stroke-opacity",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-miterlimit",
        "stroke-dasharray",
        "stroke-dashoffset",
        "opacity",
        "transform",
        "offset",
        "stop-color",
        "stop-opacity",
        "gradientUnits",
        "gradientTransform",
        "clip-path",
        "mask-type",
        "font-family",
        "font-size",
        "font-weight",
        "text-anchor",
        "dominant-baseline",
    }
)

# `style` attribute is dropped because CSS can contain expression() in
# old IE quirks and url() that exfils — easier to disallow than to
# parse. Required style settings should be inlined as proper attrs.
_BANNED_ATTR_PREFIXES = ("on",)  # onclick, onload, onmouseover, ...

# Used by clean-up regex on the raw response.
_MD_FENCE_RE = re.compile(r"^\s*```(?:xml|svg|html)?\s*\n?|\n?\s*```\s*$")
_LEADING_TEXT_RE = re.compile(r"^[^<]*?(<svg)", re.DOTALL | re.IGNORECASE)
_TRAILING_TEXT_RE = re.compile(r"(</svg>)[^<]*$", re.DOTALL | re.IGNORECASE)


def _strip_namespace(tag: str) -> str:
    """ElementTree namespaces tags as '{http://www.w3.org/2000/svg}svg' — peel."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _attr_safe(name: str, value: str) -> bool:
    """An attribute survives sanitization iff:
    - its name is in the allowlist
    - it doesn't start with an 'on' prefix (event handlers)
    - if it's a href / xlink:href, the value must start with '#' (anchor)
    """
    lower = name.lower()
    if any(lower.startswith(p) for p in _BANNED_ATTR_PREFIXES):
        return False
    if lower in ("href", "xlink:href"):
        return value.strip().startswith("#")
    return lower in {a.lower() for a in _ALLOWED_ATTRS}


def sanitize_svg(raw: str) -> str:
    """Parse `raw`, strip everything outside the allowlist, return a
    re-serialized SVG string.

    Defense in depth — even if the LLM returns malicious content (script
    tags, foreign objects, off-domain refs), the output is safe to embed
    inline in dashboard HTML.

    Raises:
        ValueError if the input doesn't contain a parseable <svg> root.
    """
    # Strip markdown fences + any prose around the SVG.
    cleaned = _MD_FENCE_RE.sub("", raw).strip()
    m_start = _LEADING_TEXT_RE.search(cleaned)
    if m_start:
        cleaned = cleaned[m_start.start(1):]
    m_end = _TRAILING_TEXT_RE.search(cleaned)
    if m_end:
        cleaned = cleaned[: m_end.end(1)]

    # LLM frequently emits `xlink:href` without declaring `xmlns:xlink`,
    # which the stdlib parser rejects as "unbound prefix". Inject the
    # declaration on the root tag so the parse succeeds; the attribute
    # itself goes through the regular allowlist check downstream.
    if "xlink:" in cleaned and "xmlns:xlink" not in cleaned:
        cleaned = re.sub(
            r"(<svg\b)",
            r'\1 xmlns:xlink="http://www.w3.org/1999/xlink"',
            cleaned,
            count=1,
        )

    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError as e:
        raise ValueError(f"failed to parse SVG: {e}") from e

    # SVG tag names are case-sensitive (`linearGradient`, not
    # `lineargradient`), so we match the allowlist case-sensitively too.
    if _strip_namespace(root.tag) != "svg":
        raise ValueError(f"root element is {root.tag!r}, expected <svg>")

    # Walk + prune in-place. ET doesn't have a built-in way to remove an
    # element from its parent without knowing the parent, so we collect
    # a (parent, child) list first.
    def walk(parent: ET.Element) -> None:
        for child in list(parent):
            tag = _strip_namespace(child.tag)
            if tag not in _ALLOWED_TAGS:
                parent.remove(child)
                continue
            # Filter attrs in place.
            for attr_name, attr_value in list(child.attrib.items()):
                if not _attr_safe(attr_name, attr_value):
                    del child.attrib[attr_name]
            walk(child)

    # Root-level attrs.
    for attr_name, attr_value in list(root.attrib.items()):
        if not _attr_safe(attr_name, attr_value):
            del root.attrib[attr_name]
    walk(root)

    # Re-emit. When the root parses without a namespace (LLM forgot to
    # set xmlns), upgrade its tag into the SVG namespace so the serializer
    # picks up the registered default xmlns automatically. This is cleaner
    # than setting `xmlns` as a literal attribute (which would double up
    # against the namespace registration when the source already had it).
    if "}" not in root.tag:
        root.tag = "{http://www.w3.org/2000/svg}svg"
    if "viewBox" not in root.attrib:
        # Fall back to a square viewBox if Claude omitted it.
        root.set("viewBox", "0 0 512 512")
    return ET.tostring(root, encoding="unicode")


# ---- rasterization (optional) ----


def is_rasterization_available() -> bool:
    """True iff cairosvg can be imported. The dashboard surfaces this so the
    operator knows whether AI-generated logos will produce a PNG raster on
    this machine, or just an SVG."""
    try:
        import importlib.util

        return importlib.util.find_spec("cairosvg") is not None
    except Exception:
        return False


def rasterize_svg_to_png(
    svg_str: str, *, output_path: Path, size_px: int = 1024
) -> bool:
    """Write `svg_str` as a square PNG at `output_path`. Returns True on
    success, False when cairosvg isn't installed or rendering fails (the
    SVG is still useful on its own — the renderer falls back to no logo
    overlay when the PNG is missing)."""
    try:
        import cairosvg  # type: ignore[import-not-found]
    except ImportError:
        _log.info(
            "logo.rasterize_skipped",
            reason="cairosvg not installed",
        )
        return False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2png(
            bytestring=svg_str.encode("utf-8"),
            write_to=str(output_path),
            output_width=size_px,
            output_height=size_px,
        )
        return True
    except Exception as e:
        _log.warning("logo.rasterize_failed", error=str(e))
        return False


# ---- prompt ----


_LOGO_SYSTEM_PROMPT = """\
You are designing a minimal SVG logo for a streamer brand.

Hard requirements:
- Output a single self-contained SVG, 512x512 viewBox.
- Transparent background. The SVG should look correct over both light
  and dark gameplay footage.
- Use ONLY the two brand colors plus #FFFFFF / transparent.
- Single bold mark — a monogram, a geometric shape, or a stylized
  icon. No multi-element scenes.
- Readable at 48px (the size it'll appear in the corner of a 1080p clip).
- Do NOT include text unless the brand name is ≤ 4 characters.
- No <script>, no <foreignObject>, no animations.
- Stick to: path, rect, circle, ellipse, polygon, polyline, line, g,
  defs, linearGradient, radialGradient. Nothing exotic.

The 'svg' field of your response must contain ONLY the raw SVG code
(<svg ...> ... </svg>). No markdown fences, no commentary.
The 'style_hint' field is a short human-readable note (≤ 120 chars)
describing the visual approach you took — useful for the operator
to remember which generation they chose.
"""


# ---- public entry: generate a logo ----


async def generate_logo(
    *,
    tenant_id: str,
    brand_name: str,
    primary_color: str,
    accent_color: str,
    style: str = "Minimal mark / monogram",
    router: LLMRouter,
) -> LogoSVG:
    """Call the LLM router, sanitize the result, return a validated LogoSVG.

    The caller persists the SVG and (optionally) rasterizes it. Raises
    LLMError on any provider failure — the dashboard surfaces the
    error inline.
    """
    user_prompt = (
        f'Brand name: "{brand_name}"\n'
        f"Primary color: {primary_color}\n"
        f"Accent color: {accent_color}\n"
        f"Style guidance: {style}\n"
    )
    response = await router.complete(
        tenant_id=tenant_id,
        purpose="logo_generation",
        system=_LOGO_SYSTEM_PROMPT,
        user=user_prompt,
        schema=LogoSVG,
    )
    safe_svg = sanitize_svg(response.svg)
    return LogoSVG(svg=safe_svg, style_hint=response.style_hint)
