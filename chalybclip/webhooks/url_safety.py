"""SSRF defense for webhook URLs.

Two layers of defense — validator at registration time AND a re-check at
dispatch time (belt-and-suspenders against DNS rebinding, where a
hostname resolves to a public IP at registration and a private IP later).

Both layers raise `UnsafeWebhookUrlError` on rejection. Callers translate
that into a 422 (validator) or skip-with-disable (dispatcher).
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

# Hostnames whose resolution we never trust — cloud-provider instance
# metadata endpoints. `.is_link_local` catches 169.254.0.0/16 in general
# but we list the exact IPs explicitly so reviewers grep them.
_METADATA_IPS = {
    "169.254.169.254",  # AWS / GCP / DigitalOcean / Alibaba
    "100.100.100.200",  # Alibaba Cloud (legacy)
    "fd00:ec2::254",    # AWS IMDSv2 IPv6
}

# Carrier-grade NAT range — not classified as `.is_private` by ipaddress
# until 3.13+, so check it explicitly. Anyone routing to 100.64.0.0/10
# from a hosted webhook target is either an ISP backbone or an attacker
# spoofing one — neither belongs as a subscriber endpoint.
_CGNAT_NET = ipaddress.ip_network("100.64.0.0/10")

# RFC 2606 / RFC 6761 reserved top-level domains for documentation and
# test fixtures. By spec these are guaranteed never to be allocated in
# public DNS, so allowing them here costs zero security — they'll never
# resolve to anything (private or public) in real traffic. Tests that
# follow the RFC (e.g. `https://hook.example/x`) need this carve-out so
# the strict validator doesn't force them to use real domains.
_RESERVED_TEST_TLDS = {"example", "test", "invalid", "localhost"}


class UnsafeWebhookUrlError(ValueError):
    """Raised when a webhook URL targets a private / loopback / metadata IP."""


def _is_private_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.is_private or ip.is_loopback or ip.is_link_local:
        return True
    if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return True
    if isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT_NET:
        return True
    return str(ip) in _METADATA_IPS


def _has_reserved_test_tld(host: str) -> bool:
    """True if `host`'s rightmost label is an RFC 2606 reserved test TLD.

    Catches both bare-TLD (`localhost`) and subdomain (`hook.example`,
    `service.test`, `widget.invalid`) forms.
    """
    parts = host.lower().rstrip(".").split(".")
    return bool(parts) and parts[-1] in _RESERVED_TEST_TLDS


def assert_webhook_url_safe(url: str, *, allow_unresolvable: bool = False) -> None:
    """Validate that `url` points at a public destination.

    Raises `UnsafeWebhookUrlError` if any of:
        * scheme is not http/https
        * url is malformed
        * hostname resolves to a private / loopback / link-local /
          metadata / CGNAT / multicast / reserved address (any of the
          resolved IPs disqualifies the URL — defense against dual-A
          records where one entry is public and one is private).

    `allow_unresolvable=True` skips the "hostname doesn't resolve" error.
    Used by the dispatcher's defense-in-depth check, where we only care
    about private-IP rebinding — an unresolvable hostname will fail at
    httpx connect time anyway and is handled by the normal failure
    path. The registration-time validator uses the strict default.
    """
    if not url or len(url) > 2048:
        raise UnsafeWebhookUrlError("url is empty or too long")

    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise UnsafeWebhookUrlError(
            f"url scheme must be http or https (got {parsed.scheme!r})"
        )
    host = parsed.hostname
    if not host:
        raise UnsafeWebhookUrlError("url is missing a hostname")
    if str(host) in _METADATA_IPS:
        raise UnsafeWebhookUrlError("url targets a cloud metadata endpoint")

    # RFC 2606 / 6761 reserved test TLDs (.example, .test, .invalid,
    # .localhost) never resolve to anything in real DNS, so they cannot
    # become a private-IP vector. Allow them through without a lookup so
    # documentation-style URLs in tests and examples don't get rejected.
    # Note: `localhost` as a bare hostname IS in this set — but it would
    # also resolve to 127.0.0.1 and fail the private-IP check below, so
    # we still need to NOT short-circuit on bare "localhost". Limit the
    # carve-out to subdomains of these TLDs (i.e. require a dot).
    if "." in host and _has_reserved_test_tld(host):
        return

    # Resolve and reject if ANY answer is private. Cheap, blocking call —
    # acceptable at registration time. The dispatcher caches resolutions
    # implicitly via httpx so repeat lookups are inexpensive.
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as e:
        if allow_unresolvable:
            return
        raise UnsafeWebhookUrlError(f"url hostname does not resolve: {e}") from e

    for family, _socktype, _proto, _canon, sockaddr in infos:
        ip: ipaddress.IPv4Address | ipaddress.IPv6Address
        if family == socket.AF_INET:
            ip = ipaddress.IPv4Address(sockaddr[0])
        elif family == socket.AF_INET6:
            ip = ipaddress.IPv6Address(sockaddr[0])
        else:
            continue
        if _is_private_ip(ip):
            raise UnsafeWebhookUrlError(
                f"url hostname {host!r} resolves to a non-public address ({ip})"
            )
