"""Links and redirects from Clips back to the Chalyb hub, in the person's language.

The hub serves English under /en (next-intl `as-needed`) and Spanish with no
prefix. When the engine knows the language (the hub's SSO `lang`, kept in the
`chalybclip_lang` cookie), hub pages go to /en/…; the hub's /auth/* launcher
sits outside its locale tree, so it gets `?lang=en` instead. Spanish or an
unknown language leaves every URL exactly as before.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from fastapi import Request

from chalybclip.api.i18n import explicit_locale

# Hub paths that are not locale-prefixed pages.
_UNPREFIXED = ("/auth/", "/api/")
_LOCALE_PREFIXES = ("/en", "/es")


def request_hub_lang(request: Request) -> str | None:
    """The language the person explicitly chose (`?lang=` or the SSO cookie)."""
    if not hasattr(request, "query_params"):  # gates called with a bare state stub
        return None
    return explicit_locale(request)


def hub_url(url: str, lang: str | None) -> str:
    """`url` (absolute hub URL or a bare hub path) in `lang`."""
    if lang != "en":
        return url
    parts = urlsplit(url)
    path = parts.path or "/"
    if any(path == p or path.startswith(f"{p}/") for p in _LOCALE_PREFIXES):
        return url
    if path.startswith(_UNPREFIXED):
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "lang"]
        query.append(("lang", "en"))
        return urlunsplit(parts._replace(query=urlencode(query)))
    new_path = "/en" if path == "/" else f"/en{path}"
    return urlunsplit(parts._replace(path=new_path))


__all__ = ["hub_url", "request_hub_lang"]
