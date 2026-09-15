"""Zernio inbound webhook signature verification + parsing tests."""
from __future__ import annotations

import hashlib
import hmac

from chalybclip.integrations.zernio.webhooks import (
    parse_post_event,
    verify_zernio_signature,
)


def _sign(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_verify_accepts_valid_hex_signature() -> None:
    body = b'{"event":"post.published","post":{"_id":"p1"}}'
    sig = _sign("whsec", body)
    assert verify_zernio_signature(secret="whsec", body=body, signature_header=sig)


def test_verify_accepts_sha256_prefixed_signature() -> None:
    body = b'{"x":1}'
    sig = "sha256=" + _sign("whsec", body)
    assert verify_zernio_signature(secret="whsec", body=body, signature_header=sig)


def test_verify_rejects_tampered_signature() -> None:
    body = b'{"x":1}'
    good = _sign("whsec", body)
    bad = good[:-1] + ("0" if good[-1] != "0" else "1")
    assert not verify_zernio_signature(secret="whsec", body=body, signature_header=bad)


def test_verify_rejects_wrong_secret() -> None:
    body = b'{"x":1}'
    sig = _sign("other", body)
    assert not verify_zernio_signature(secret="whsec", body=body, signature_header=sig)


def test_verify_rejects_missing_inputs() -> None:
    assert not verify_zernio_signature(secret="", body=b"x", signature_header="abc")
    assert not verify_zernio_signature(secret="whsec", body=b"x", signature_header=None)


def test_parse_post_event_flattens_nested_post() -> None:
    out = parse_post_event(
        {
            "event": "post.published",
            "post": {"_id": "p1", "status": "published", "profileId": "ten_alice"},
        }
    )
    assert out["event"] == "post.published"
    assert out["post_id"] == "p1"
    assert out["status"] == "published"
    assert out["profile_id"] == "ten_alice"


def test_parse_post_event_handles_flat_payload() -> None:
    out = parse_post_event({"type": "post.failed", "id": "p2", "status": "failed"})
    assert out["event"] == "post.failed"
    assert out["post_id"] == "p2"
    assert out["status"] == "failed"


def test_verify_rejects_non_ascii_signature_without_raising() -> None:
    """Regression: `hmac.compare_digest` on str raises TypeError for
    non-ASCII input — a hostile header on the public webhook route must
    map to False (→ 403), never an unhandled 500."""
    assert not verify_zernio_signature(
        secret="whsec", body=b'{"x":1}', signature_header="ñdeadbeef"
    )
