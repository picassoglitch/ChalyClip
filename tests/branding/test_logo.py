"""AI logo generation — SVG sanitization + Anthropic round-trip (slice D.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from chalybclip.branding import (
    LogoSVG,
    generate_logo,
    is_rasterization_available,
    rasterize_svg_to_png,
    sanitize_svg,
)
from chalybclip.llm import LLMRouter
from chalybclip.llm.config import ProviderConfig
from tests.llm._fakes import FakeProvider  # type: ignore[import]
from tests.llm._fixtures import make_llm_config  # type: ignore[import]

# ---- sanitize_svg ----


def test_sanitize_passes_a_clean_minimal_svg() -> None:
    raw = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<circle cx="50" cy="50" r="40" fill="#FF3366"/>'
        "</svg>"
    )
    out = sanitize_svg(raw)
    assert "<svg" in out
    assert 'cx="50"' in out
    assert 'fill="#FF3366"' in out


def test_sanitize_strips_script_tag() -> None:
    """A <script> inside the SVG must not survive sanitization."""
    raw = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<script>alert(1)</script>'
        '<circle cx="50" cy="50" r="40" fill="#000"/>'
        "</svg>"
    )
    out = sanitize_svg(raw)
    assert "script" not in out.lower()
    assert "alert" not in out


def test_sanitize_strips_event_handler_attrs() -> None:
    raw = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<circle cx="50" cy="50" r="40" fill="#000" onclick="bad()"/>'
        "</svg>"
    )
    out = sanitize_svg(raw)
    assert "onclick" not in out
    assert "bad()" not in out


def test_sanitize_drops_unknown_elements() -> None:
    raw = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<foreignObject><div>html</div></foreignObject>'
        '<rect width="50" height="50" fill="#0F0"/>'
        "</svg>"
    )
    out = sanitize_svg(raw)
    assert "foreignObject" not in out
    assert "<div" not in out
    assert "<rect" in out


def test_sanitize_keeps_gradient_and_path() -> None:
    raw = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
        '<defs>'
        '<linearGradient id="g1"><stop offset="0" stop-color="#FF3366"/>'
        '<stop offset="1" stop-color="#FFD700"/></linearGradient>'
        '</defs>'
        '<path d="M10 10 L90 90 Z" fill="url(#g1)"/>'
        "</svg>"
    )
    out = sanitize_svg(raw)
    assert "linearGradient" in out
    assert "<path" in out
    assert "stop-color" in out


def test_sanitize_strips_markdown_fence() -> None:
    """The LLM sometimes returns ```xml ... ``` — the cleaner must
    peel it off before parsing."""
    raw = (
        "```xml\n"
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<rect width="10" height="10"/>'
        "</svg>\n"
        "```"
    )
    out = sanitize_svg(raw)
    assert "<svg" in out
    assert "```" not in out


def test_sanitize_strips_prose_before_and_after() -> None:
    raw = (
        "Here is the logo you requested:\n"
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<rect width="10" height="10"/>'
        "</svg>\n"
        "Let me know if you want a variant!"
    )
    out = sanitize_svg(raw)
    assert "Here is" not in out
    assert "variant" not in out
    assert "<svg" in out


def test_sanitize_rejects_non_svg_root() -> None:
    raw = "<html><body>nope</body></html>"
    with pytest.raises(ValueError):
        sanitize_svg(raw)


def test_sanitize_rejects_unparseable() -> None:
    raw = "this isn't xml at all <broken"
    with pytest.raises(ValueError):
        sanitize_svg(raw)


def test_sanitize_keeps_xlink_href_to_anchor_only() -> None:
    """`<use xlink:href="#id">` is allowed; remote `<use xlink:href="http://...">` is not."""
    safe_raw = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<defs><circle id="dot" r="2"/></defs>'
        '<use xlink:href="#dot"/>'
        "</svg>"
    )
    out = sanitize_svg(safe_raw)
    assert "use" in out

    bad_raw = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<use xlink:href="http://evil.example/x.svg"/>'
        "</svg>"
    )
    out2 = sanitize_svg(bad_raw)
    # The <use> element itself can stay, but the off-domain href is gone.
    assert "http://evil" not in out2


def test_sanitize_adds_default_viewbox_when_missing() -> None:
    raw = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<rect width="10" height="10"/>'
        "</svg>"
    )
    out = sanitize_svg(raw)
    assert "viewBox" in out


def test_sanitize_adds_xmlns_when_missing() -> None:
    raw = '<svg viewBox="0 0 10 10"><rect width="10" height="10"/></svg>'
    out = sanitize_svg(raw)
    assert "http://www.w3.org/2000/svg" in out


# ---- rasterize_svg_to_png ----


def test_rasterize_skips_gracefully_when_unavailable(tmp_path: Path) -> None:
    """If cairosvg isn't installed (typical dev machine), rasterize returns
    False and the SVG-only output is still useful."""
    if is_rasterization_available():
        pytest.skip("cairosvg is installed — skip the not-available path test")
    raw = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10">'
        '<rect width="10" height="10" fill="#000"/>'
        "</svg>"
    )
    ok = rasterize_svg_to_png(raw, output_path=tmp_path / "logo.png", size_px=64)
    assert ok is False
    assert not (tmp_path / "logo.png").exists()


# ---- generate_logo ----


async def test_generate_logo_round_trip_through_fake_router() -> None:
    """generate_logo → fake provider → sanitized output is what we asked
    for. End-to-end through the real router minus the network."""
    provider = FakeProvider("anthropic")
    provider.queue_success(
        {
            "svg": (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
                '<circle cx="256" cy="256" r="200" fill="#FF3366"/>'
                "</svg>"
            ),
            "style_hint": "Bold circle on transparent.",
        }
    )

    def factory(name: str, _config: ProviderConfig, _api_key: str):
        return provider if name == "anthropic" else None

    config = make_llm_config(
        retry_attempts=1,
        purpose="logo_generation",
    )
    router = LLMRouter(
        config=config,
        api_keys={"anthropic": "k"},
        provider_factory=factory,
    )
    out = await generate_logo(
        tenant_id="default",
        brand_name="AARA",
        primary_color="#FF3366",
        accent_color="#FFD700",
        router=router,
    )
    assert isinstance(out, LogoSVG)
    assert "<svg" in out.svg
    assert "#FF3366" in out.svg
    assert out.style_hint == "Bold circle on transparent."
    # Provider received the right purpose / brand context.
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert "AARA" in call["user"]
    assert "#FF3366" in call["user"]


async def test_generate_logo_sanitizes_a_malicious_response() -> None:
    """Even when the LLM returns a script tag the public API surface
    must be safe to inline into the dashboard."""
    provider = FakeProvider("anthropic")
    provider.queue_success(
        {
            "svg": (
                '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
                "<script>alert('pwn')</script>"
                '<rect width="512" height="512" fill="#000"/>'
                "</svg>"
            ),
            "style_hint": "Square with a hidden surprise.",
        }
    )

    def factory(name: str, _config: ProviderConfig, _api_key: str):
        return provider if name == "anthropic" else None

    router = LLMRouter(
        config=make_llm_config(retry_attempts=1, purpose="logo_generation"),
        api_keys={"anthropic": "k"},
        provider_factory=factory,
    )
    out = await generate_logo(
        tenant_id="default",
        brand_name="X",
        primary_color="#000000",
        accent_color="#FFFFFF",
        router=router,
    )
    assert "<script" not in out.svg
    assert "pwn" not in out.svg
    assert "<rect" in out.svg
