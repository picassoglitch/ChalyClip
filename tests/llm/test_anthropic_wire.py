"""The real router + AnthropicProvider + SDK against a local HTTP server that
speaks the Messages API, so every outcome crosses the wire and lands in llm_calls."""

from __future__ import annotations

import asyncio
import base64
import json
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from chalybclip.branding.logo import generate_logo
from chalybclip.db import Database, LLMCallsRepo, TenantsRepo, apply_migrations
from chalybclip.db.usage_repos import UsageOutboxRepo
from chalybclip.detect.viral import ViralMomentList, _format_for_llm
from chalybclip.errors import LLMError
from chalybclip.llm import (
    CropBoxVerdict,
    FaceEmotionVerdict,
    LLMRouter,
    MultimodalImage,
    RescoreVerdict,
    ThumbnailPickVerdict,
)
from chalybclip.llm.config import LLMConfig, load_llm_config
from chalybclip.llm.router import reset_billing_lockouts
from chalybclip.tenancy import bound_tenant
from chalybclip.transcribe.models import Transcript
from chalybclip.variants.hooks import HookBatch, generate_hooks

_EXAMPLE = Path(__file__).resolve().parents[2] / "config" / "llm.example.yaml"
TENANT = "ten_a"


# ---- fake Messages API ---------------------------------------------------


@dataclass
class Reply:
    status: int = 200
    body: dict[str, Any] = field(default_factory=dict)
    delay_s: float = 0.0
    headers: dict[str, str] = field(default_factory=dict)


class FakeAnthropic:
    def __init__(self) -> None:
        self.script: list[Reply] = []
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self._lock = threading.Lock()

    async def messages(self, request: Request) -> JSONResponse:
        body = await request.json()
        with self._lock:
            self.requests.append(body)
            self.headers.append(dict(request.headers))
            reply = self.script.pop(0) if self.script else None
        if reply is None:
            return JSONResponse(_error("api_error", "unscripted request"), status_code=500)
        if reply.delay_s:
            await asyncio.sleep(reply.delay_s)
        return JSONResponse(reply.body, status_code=reply.status, headers=reply.headers)


def message(
    model: str,
    payload: Any,
    *,
    stop: str = "end_turn",
    input_tokens: int = 1_000,
    output_tokens: int = 200,
    cache_read: int = 0,
    cache_write: int = 0,
    category: str | None = None,
    raw: bool = False,
) -> Reply:
    """A Messages API reply: an (empty) thinking block, then the JSON text block."""
    content: list[dict[str, Any]] = [{"type": "thinking", "thinking": "", "signature": "sig"}]
    if payload is not None:
        text = payload if raw else json.dumps(payload, ensure_ascii=False)
        content.append({"type": "text", "text": text})
    details = (
        {"type": "refusal", "category": category, "explanation": "declined"}
        if stop == "refusal"
        else None
    )
    return Reply(
        body={
            "id": "msg_01Test",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "stop_details": details,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
                "service_tier": "standard",
            },
        }
    )


def _error(kind: str, msg: str) -> dict[str, Any]:
    return {"type": "error", "error": {"type": kind, "message": msg}, "request_id": "req_1"}


def error(status: int, kind: str, msg: str = "boom", **headers: str) -> Reply:
    return Reply(status=status, body=_error(kind, msg), headers=headers)


@pytest.fixture(scope="module")
def server() -> Iterator[tuple[FakeAnthropic, str]]:
    fake = FakeAnthropic()
    app = Starlette(routes=[Route("/v1/messages", fake.messages, methods=["POST"])])
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        yield fake, f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True
        thread.join(timeout=5)


@pytest.fixture
def api(server: tuple[FakeAnthropic, str]) -> Iterator[FakeAnthropic]:
    fake, _ = server
    fake.script.clear()
    fake.requests.clear()
    fake.headers.clear()
    reset_billing_lockouts()
    yield fake
    reset_billing_lockouts()
    assert not fake.script, "scripted replies left unused"


@pytest.fixture
async def db(tmp_path: Path) -> Any:
    d = Database(tmp_path / "t.db")
    await apply_migrations(d)
    await TenantsRepo(d).create(tenant_id=TENANT, name="A")
    try:
        yield d
    finally:
        await d.close()


def committed_config(
    base_url: str,
    *,
    timeout_s: float | None = None,
    standard: str | None = None,
    premium: str | None = None,
) -> LLMConfig:
    """The committed llm.example.yaml, pointed at the fake server, no backoff sleeps."""
    cfg = load_llm_config(_EXAMPLE)
    anth = cfg.providers["anthropic"]
    models = anth.models.model_copy(
        update={k: v for k, v in {"standard": standard, "premium": premium}.items() if v}
    )
    update: dict[str, Any] = {"base_url": base_url, "models": models}
    if timeout_s is not None:
        update["timeout_s"] = timeout_s
    providers = {**cfg.providers, "anthropic": anth.model_copy(update=update)}
    retry = cfg.retry.model_copy(update={"initial_backoff_s": 0.0})
    return cfg.model_copy(update={"providers": providers, "retry": retry})


def make_router(base_url: str, db: Database, **kw: Any) -> LLMRouter:
    return LLMRouter(committed_config(base_url, **kw), api_keys={"anthropic": "sk-test"}, db=db)


async def rows(db: Database) -> list[Any]:
    with bound_tenant(TENANT):
        return await LLMCallsRepo(db).list_for_tenant()


def haiku55(inp: int, out: int) -> int:
    return round(inp * 0.10 + out * 0.50)


@pytest.fixture
def url(server: tuple[FakeAnthropic, str]) -> str:
    return server[1]


# ---- happy path per routed purpose ---------------------------------------

ES = "¿Qué pasó? ¡Increíble jugada de Ñoño! 🔥🎮 — 「最高」"
HOOKS = {"hooks": [{"text": "¡Nadie esperaba esto! 😱"}, {"text": "Ñoño se lució 🔥"}]}
VIRAL = {
    "moments": [
        {
            "timestamp_s": 12.5,
            "duration_s": 24.0,
            "score": 0.91,
            "reason": "Reacción épica con acentos: canción",
            "type": "humor",
            "transcript_snippet": "¡No manches, qué golazo! 🤯",
        }
    ]
}
SVG = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><circle cx="5" cy="5" r="4"/></svg>'


async def test_hook_generation_end_to_end_spanish_and_emoji(api, url, db) -> None:
    api.script.append(message("claude-haiku-5-5", HOOKS, input_tokens=812, output_tokens=95))
    router = make_router(url, db)
    with bound_tenant(TENANT):
        hooks = await generate_hooks(
            tenant_id=TENANT,
            persona_voice=ES,
            persona_language="es",
            transcript_snippet=ES,
            router=router,
        )
    assert [h.text for h in hooks] == ["¡Nadie esperaba esto! 😱", "Ñoño se lució 🔥"]
    req = api.requests[0]
    assert req["model"] == "claude-haiku-5-5"
    assert "tools" not in req and "tool_choice" not in req
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert "effort" not in req["output_config"]  # hook_generation keeps the model default
    assert req["max_tokens"] == 16000
    assert ES in req["messages"][0]["content"]  # UTF-8 survived the wire
    assert api.headers[0].get("x-stainless-retry-count") == "0"
    (row,) = await rows(db)
    assert (row.purpose, row.model, row.status, row.attempts) == (
        "hook_generation",
        "claude-haiku-5-5",
        "ok",
        1,
    )
    assert (row.input_tokens, row.output_tokens) == (812, 95)
    assert row.cost_usd_micros == haiku55(812, 95)


async def test_viral_detection_sends_effort_low_and_bills_haiku_5_5(api, url, db) -> None:
    api.script.append(message("claude-haiku-5-5", VIRAL, input_tokens=40_000, output_tokens=900))
    router = make_router(url, db)
    with bound_tenant(TENANT):
        out = await router.complete(
            tenant_id=TENANT,
            purpose="viral_detection",
            system="s",
            user=ES,
            schema=ViralMomentList,
        )
    assert out.moments[0].transcript_snippet == "¡No manches, qué golazo! 🤯"
    assert api.requests[0]["output_config"]["effort"] == "low"
    (row,) = await rows(db)
    assert row.cost_usd_micros == haiku55(40_000, 900) == 4_450


async def test_viral_detection_on_an_empty_transcript_sends_a_placeholder(api, url, db) -> None:
    empty = Transcript(stream_id="str_1", tenant_id=TENANT, language="es", duration_s=0, model="m")
    user = _format_for_llm(empty, duration_s=0)
    assert user.strip()  # the API 400s on an empty text block
    api.script.append(
        message("claude-haiku-5-5", {"moments": []}, input_tokens=300, output_tokens=8)
    )
    router = make_router(url, db)
    with bound_tenant(TENANT):
        out = await router.complete(
            tenant_id=TENANT,
            purpose="viral_detection",
            system="s",
            user=user,
            schema=ViralMomentList,
        )
    assert out.moments == []
    assert api.requests[0]["messages"][0]["content"] == user


async def test_logo_generation_routes_premium_to_opus_5_5(api, url, db) -> None:
    api.script.append(
        message(
            "claude-opus-5-5",
            {"svg": SVG, "style_hint": "Monograma ñ"},
            input_tokens=1_500,
            output_tokens=3_000,
            cache_read=10_000,
        )
    )
    router = make_router(url, db)
    with bound_tenant(TENANT):
        logo = await generate_logo(
            tenant_id=TENANT,
            brand_name="Ñandú 🎮",
            primary_color="#000",
            accent_color="#fff",
            router=router,
        )
    assert "<circle" in logo.svg and logo.style_hint == "Monograma ñ"
    req = api.requests[0]
    assert req["model"] == "claude-opus-5-5"
    assert "effort" not in req["output_config"]  # premium creative purpose: model default
    (row,) = await rows(db)
    assert row.quality == "premium"
    # $4 in / $20 out / $0.20 cache read.
    assert row.cost_usd_micros == 1_500 * 4 + 3_000 * 20 + 10_000 * 0.2 == 68_000


@pytest.mark.parametrize(
    ("schema", "payload"),
    [
        (RescoreVerdict, {"score": 0.8, "face_emotion": "laugh", "reason": "Risa genuina 😂"}),
        (CropBoxVerdict, {"x_frac": 0.25, "width_frac": 0.5625, "reason": "cara centrada"}),
        (ThumbnailPickVerdict, {"index": 2, "reason": "expresión más fuerte"}),
        (FaceEmotionVerdict, {"has_face": True, "emotion": "shock", "confidence": 0.7}),
    ],
)
async def test_vision_rescore_schemas_cross_the_wire(api, url, db, schema, payload) -> None:
    api.script.append(message("claude-haiku-5-5", payload, input_tokens=3_200, output_tokens=60))
    router = make_router(url, db)
    jpeg = b"\xff\xd8\xff\xe0fake-jpeg"
    with bound_tenant(TENANT):
        out = await router.complete_multimodal(
            tenant_id=TENANT,
            purpose="vision_rescore",
            system="s",
            user="califica",
            images=[MultimodalImage(data=jpeg), MultimodalImage(data=jpeg)],
            schema=schema,
        )
    assert out.model_dump(exclude_unset=True).items() >= payload.items()
    req = api.requests[0]
    blocks = req["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["image", "image", "text"]
    assert blocks[0]["source"]["data"] == base64.b64encode(jpeg).decode()
    assert req["output_config"]["effort"] == "low"
    assert req["output_config"]["format"]["schema"]["additionalProperties"] is False
    (row,) = await rows(db)
    assert row.cost_usd_micros == haiku55(3_200, 60)


@pytest.mark.parametrize(
    ("purpose", "model", "effort"),
    [
        ("clip_scoring", "claude-haiku-5-5", "low"),
        ("variant_generation_visual", "claude-haiku-5-5", None),
        ("agent_decision", "claude-opus-5-5", None),
    ],
)
async def test_config_only_purposes_route_and_validate(
    api, url, db, purpose, model, effort
) -> None:
    api.script.append(message(model, {"score": 0.5, "reason": "ok"}, output_tokens=10))
    router = make_router(url, db)
    with bound_tenant(TENANT):
        await router.complete(
            tenant_id=TENANT, purpose=purpose, system="s", user="u", schema=RescoreVerdict
        )
    req = api.requests[0]
    assert req["model"] == model
    assert req["output_config"].get("effort") == effort


# ---- boundaries: Haiku 5.5's >100K prompt tier ----------------------------


@pytest.mark.parametrize(
    ("inp", "cache_read", "expected"),
    [
        (100_000, 0, round(100_000 * 0.10 + 1_000 * 0.50)),  # at the edge: base tier
        (100_001, 0, round(100_001 * 0.50 + 1_000 * 2.50)),  # one over: every token long
        (20_000, 90_000, round(20_000 * 0.50 + 90_000 * 0.05 + 1_000 * 2.50)),  # cache counts
    ],
)
async def test_haiku_5_5_long_prompt_tier_over_the_wire(
    api, url, db, inp, cache_read, expected
) -> None:
    api.script.append(
        message(
            "claude-haiku-5-5",
            {"moments": []},
            input_tokens=inp,
            output_tokens=1_000,
            cache_read=cache_read,
        )
    )
    router = make_router(url, db)
    with bound_tenant(TENANT):
        await router.complete(
            tenant_id=TENANT,
            purpose="viral_detection",
            system="s",
            user="x" * 50,
            schema=ViralMomentList,
        )
    (row,) = await rows(db)
    assert row.cost_usd_micros == expected


async def test_retries_are_tiered_per_attempt_not_on_the_sum(api, url, db) -> None:
    # 60K + 60K = 120K summed, but each request is under 100K: both stay base tier.
    api.script += [
        message(
            "claude-haiku-5-5", "not json at all", raw=True, input_tokens=60_000, output_tokens=100
        ),
        message("claude-haiku-5-5", {"moments": []}, input_tokens=60_000, output_tokens=100),
    ]
    router = make_router(url, db)
    with bound_tenant(TENANT):
        await router.complete(
            tenant_id=TENANT,
            purpose="viral_detection",
            system="s",
            user="u",
            schema=ViralMomentList,
        )
    (row,) = await rows(db)
    assert row.attempts == 2 and row.input_tokens == 120_000
    assert row.cost_usd_micros == 2 * haiku55(60_000, 100)


# ---- bad output from the model --------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        message("claude-haiku-5-5", '{"hooks": [', raw=True, output_tokens=40),  # invalid JSON
        message("claude-haiku-5-5", '["a", "b"]', raw=True, output_tokens=40),  # not an object
        message("claude-haiku-5-5", {"hooks": []}, output_tokens=40),  # schema: min 1 hook
        message("claude-haiku-5-5", {"hooks": [{"text": "x"}], "extra": 1}, output_tokens=40),
        message("claude-haiku-5-5", None, output_tokens=40),  # only a thinking block
    ],
    ids=["invalid-json", "json-array", "schema-min", "schema-extra", "no-text-block"],
)
async def test_unusable_reply_is_retried_and_every_attempt_billed(api, url, db, bad) -> None:
    api.script += [bad, message("claude-haiku-5-5", HOOKS, output_tokens=60)]
    router = make_router(url, db)
    with bound_tenant(TENANT):
        await router.complete(
            tenant_id=TENANT, purpose="hook_generation", system="s", user="u", schema=HookBatch
        )
    assert len(api.requests) == 2
    (row,) = await rows(db)
    assert (row.status, row.attempts, row.output_tokens) == ("ok", 2, 100)
    assert row.cost_usd_micros == haiku55(1_000, 40) + haiku55(1_000, 60)


async def test_three_unusable_replies_fail_with_all_three_billed(api, url, db) -> None:
    api.script += [message("claude-haiku-5-5", "nope", raw=True, output_tokens=30)] * 3
    router = make_router(url, db)
    with bound_tenant(TENANT), pytest.raises(LLMError, match="not a JSON object"):
        await router.complete(
            tenant_id=TENANT, purpose="hook_generation", system="s", user="u", schema=HookBatch
        )
    (row,) = await rows(db)
    assert (row.status, row.attempts) == ("error", 3)
    assert row.cost_usd_micros == 3 * haiku55(1_000, 30)
    ev = (await UsageOutboxRepo(db).get(row.id)).payload
    assert ev["cost_usd_micros"] == row.cost_usd_micros


async def test_refusal_is_billed_once_and_not_retried(api, url, db) -> None:
    api.script.append(
        message(
            "claude-opus-5-5",
            None,
            stop="refusal",
            category="cyber",
            input_tokens=900,
            output_tokens=12,
        )
    )
    router = make_router(url, db)
    with bound_tenant(TENANT), pytest.raises(LLMError, match=r"RefusalError.*category=cyber"):
        await generate_logo(
            tenant_id=TENANT,
            brand_name="x",
            primary_color="#000",
            accent_color="#fff",
            router=router,
        )
    assert len(api.requests) == 1
    (row,) = await rows(db)
    assert (row.status, row.attempts, row.model) == ("error", 1, "claude-opus-5-5")
    assert row.cost_usd_micros == 900 * 4 + 12 * 20


async def test_max_tokens_truncation_is_billed_once_and_not_retried(api, url, db) -> None:
    api.script.append(
        message(
            "claude-haiku-5-5",
            '{"moments": [{"timestamp_s": 1',
            raw=True,
            stop="max_tokens",
            input_tokens=5_000,
            output_tokens=16_000,
        )
    )
    router = make_router(url, db)
    with bound_tenant(TENANT), pytest.raises(LLMError, match="TruncatedOutputError"):
        await router.complete(
            tenant_id=TENANT,
            purpose="viral_detection",
            system="s",
            user="u",
            schema=ViralMomentList,
        )
    assert len(api.requests) == 1
    (row,) = await rows(db)
    assert (row.status, row.attempts, row.output_tokens) == ("error", 1, 16_000)
    assert row.cost_usd_micros == haiku55(5_000, 16_000)


# ---- dependencies down -----------------------------------------------------


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (408, "timeout_error"),
        (409, "conflict_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (502, "api_error"),
        (503, "api_error"),
        (504, "api_error"),
        (529, "overloaded_error"),
    ],
)
async def test_transient_status_is_retried_by_the_router_only(api, url, db, status, kind) -> None:
    api.script += [
        error(status, kind, **{"retry-after": "0", "x-should-retry": "true"}),
        message("claude-haiku-5-5", HOOKS, output_tokens=50),
    ]
    router = make_router(url, db)
    with bound_tenant(TENANT):
        await router.complete(
            tenant_id=TENANT, purpose="hook_generation", system="s", user="u", schema=HookBatch
        )
    assert len(api.requests) == 2  # one router retry, zero SDK retries
    (row,) = await rows(db)
    assert (row.status, row.attempts) == ("ok", 2)
    assert row.cost_usd_micros == haiku55(1_000, 50)  # the error attempt had no usage


async def test_three_overloads_exhaust_and_log_a_zero_cost_error(api, url, db) -> None:
    api.script += [error(529, "overloaded_error", "Overloaded")] * 3
    router = make_router(url, db)
    with bound_tenant(TENANT), pytest.raises(LLMError, match="529"):
        await router.complete(
            tenant_id=TENANT, purpose="hook_generation", system="s", user="u", schema=HookBatch
        )
    assert len(api.requests) == 3
    (row,) = await rows(db)
    assert (row.status, row.attempts, row.cost_usd_micros) == ("error", 3, 0)


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (413, "request_too_large"),
    ],
)
async def test_client_errors_are_not_retried(api, url, db, status, kind) -> None:
    api.script.append(
        error(status, kind, "tool_choice not supported", **{"x-should-retry": "true"})
    )
    router = make_router(url, db)
    with bound_tenant(TENANT), pytest.raises(LLMError, match=str(status)):
        await router.complete(
            tenant_id=TENANT, purpose="hook_generation", system="s", user="u", schema=HookBatch
        )
    assert len(api.requests) == 1
    (row,) = await rows(db)
    assert (row.status, row.attempts) == ("error", 1)


async def test_credit_balance_400_trips_the_billing_lockout(api, url, db) -> None:
    api.script.append(error(400, "invalid_request_error", "Your credit balance is too low"))
    router = make_router(url, db)
    with bound_tenant(TENANT):
        for _ in range(2):
            with pytest.raises(LLMError):
                await router.complete(
                    tenant_id=TENANT,
                    purpose="hook_generation",
                    system="s",
                    user="u",
                    schema=HookBatch,
                )
    assert len(api.requests) == 1  # second call never left the process


async def test_connection_refused_is_retried_then_fails_loudly(db) -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    dead = f"http://127.0.0.1:{sock.getsockname()[1]}"
    sock.close()
    router = make_router(dead, db)
    with bound_tenant(TENANT), pytest.raises(LLMError, match=r"(?i)connection error"):
        await router.complete(
            tenant_id=TENANT, purpose="hook_generation", system="s", user="u", schema=HookBatch
        )
    (row,) = await rows(db)
    assert (row.status, row.attempts, row.cost_usd_micros) == ("error", 3, 0)


async def test_sdk_timeout_is_retried_until_the_standard_deadline(api, url, db) -> None:
    # timeout 0.4 s per HTTP attempt; standard deadline = 2 x 0.4 s.
    api.script += [Reply(body={}, delay_s=1.5)] * 2
    router = make_router(url, db, timeout_s=0.4)
    t0 = time.monotonic()
    with bound_tenant(TENANT), pytest.raises(LLMError, match="deadline"):
        await router.complete(
            tenant_id=TENANT, purpose="hook_generation", system="s", user="u", schema=HookBatch
        )
    assert time.monotonic() - t0 < 1.3
    assert len(api.requests) == 2
    (row,) = await rows(db)
    assert row.status == "error" and row.attempts == 2


async def test_premium_gets_one_attempt_inside_its_deadline(api, url, db) -> None:
    api.script.append(Reply(body={}, delay_s=1.5))
    router = make_router(url, db, timeout_s=0.4)
    t0 = time.monotonic()
    with bound_tenant(TENANT), pytest.raises(LLMError):
        await router.complete(
            tenant_id=TENANT, purpose="logo_generation", system="s", user="u", schema=HookBatch
        )
    assert time.monotonic() - t0 < 0.9
    assert len(api.requests) == 1


async def test_deadline_still_bills_the_attempts_that_finished(api, url, db) -> None:
    api.script += [
        message("claude-haiku-5-5", "nope", raw=True, input_tokens=2_000, output_tokens=50),
        Reply(body={}, delay_s=1.5),  # SDK timeout at 0.4 s, retried
        Reply(body={}, delay_s=1.5),  # cut by the 0.8 s router deadline
    ]
    router = make_router(url, db, timeout_s=0.4)
    with bound_tenant(TENANT), pytest.raises(LLMError, match="deadline"):
        await router.complete(
            tenant_id=TENANT, purpose="hook_generation", system="s", user="u", schema=HookBatch
        )
    (row,) = await rows(db)
    assert (row.status, row.attempts) == ("error", 3)
    assert row.cost_usd_micros == haiku55(2_000, 50)


async def test_refusal_after_a_bad_attempt_bills_both(api, url, db) -> None:
    api.script += [
        message("claude-haiku-5-5", {"hooks": []}, input_tokens=700, output_tokens=20),
        message(
            "claude-haiku-5-5",
            None,
            stop="refusal",
            category=None,
            input_tokens=700,
            output_tokens=5,
        ),
    ]
    router = make_router(url, db)
    with bound_tenant(TENANT), pytest.raises(LLMError, match="RefusalError"):
        await router.complete(
            tenant_id=TENANT, purpose="hook_generation", system="s", user="u", schema=HookBatch
        )
    assert len(api.requests) == 2
    (row,) = await rows(db)
    assert (row.status, row.attempts) == ("error", 2)
    assert row.cost_usd_micros == haiku55(700, 20) + haiku55(700, 5)


async def test_cancelled_attempt_is_flagged_as_unbilled(api, url, db, caplog) -> None:
    api.script.append(Reply(body={}, delay_s=1.5))
    router = make_router(url, db, timeout_s=0.4)
    with bound_tenant(TENANT), caplog.at_level("WARNING"), pytest.raises(LLMError):
        await router.complete(
            tenant_id=TENANT,
            purpose="logo_generation",
            system="s",
            user="u" * 400,
            schema=HookBatch,
        )
    assert "llm.unbilled_timeout" in caplog.text
    (row,) = await rows(db)
    assert row.cost_usd_micros == 0 and "in_flight_cancelled=true" in (row.error or "")


async def test_model_output_never_reaches_the_error_row(api, url, db) -> None:
    api.script += [
        message("claude-haiku-5-5", "¡secreto del stream! " * 50, raw=True),
        message("claude-haiku-5-5", {"hooks": [{"text": "¡secreto del stream!"}], "x": 1}),
        message("claude-haiku-5-5", {"hooks": [{"text": "¡secreto del stream!" * 20}]}),
    ]
    router = make_router(url, db)
    with bound_tenant(TENANT), pytest.raises(LLMError):
        await router.complete(
            tenant_id=TENANT, purpose="hook_generation", system="s", user="u", schema=HookBatch
        )
    (row,) = await rows(db)
    assert row.error and "secreto" not in row.error  # CLAUDE.md rule 10


async def test_legacy_retryable_list_in_yaml_stops_retrying_overloads(api, url, db) -> None:
    # An llm.yaml still carrying the pre-change list: 529 then fails on the first attempt.
    cfg = committed_config(url)
    cfg = cfg.model_copy(
        update={
            "retry": cfg.retry.model_copy(
                update={"retryable_status_codes": [429, 500, 502, 503, 504]}
            )
        }
    )
    api.script.append(error(529, "overloaded_error", "Overloaded"))
    router = LLMRouter(cfg, api_keys={"anthropic": "sk-test"}, db=db)
    with bound_tenant(TENANT), pytest.raises(LLMError, match="529"):
        await router.complete(
            tenant_id=TENANT, purpose="hook_generation", system="s", user="u", schema=HookBatch
        )
    assert len(api.requests) == 1


# ---- state: concurrent calls ----------------------------------------------


async def test_concurrent_calls_each_get_their_own_row(api, url, db) -> None:
    n = 12
    api.script += [
        message("claude-haiku-5-5", HOOKS, input_tokens=100 + i, output_tokens=10) for i in range(n)
    ]
    for i in range(0, n, 3):  # a few replies arrive late to interleave the calls
        api.script[i].delay_s = 0.2
    router = make_router(url, db)
    with bound_tenant(TENANT):
        results = await asyncio.gather(
            *[
                router.complete(
                    tenant_id=TENANT,
                    purpose="hook_generation",
                    system="s",
                    user=f"u{i}",
                    schema=HookBatch,
                )
                for i in range(n)
            ]
        )
    assert len(results) == n and len(api.requests) == n
    got = await rows(db)
    assert len(got) == n and all(r.status == "ok" for r in got)
    assert sorted(r.input_tokens for r in got) == [100 + i for i in range(n)]
    assert sum(r.cost_usd_micros for r in got) == sum(haiku55(100 + i, 10) for i in range(n))


# ---- config rollback to haiku-4-5 / opus-5 --------------------------------


async def test_rollback_to_haiku_4_5_drops_effort_and_uses_yaml_prices(api, url, db) -> None:
    api.script.append(message("claude-haiku-4-5", VIRAL, input_tokens=150_000, output_tokens=900))
    router = make_router(url, db, standard="claude-haiku-4-5", premium="claude-opus-5")
    with bound_tenant(TENANT):
        await router.complete(
            tenant_id=TENANT,
            purpose="viral_detection",
            system="s",
            user="u",
            schema=ViralMomentList,
        )
    req = api.requests[0]
    assert req["model"] == "claude-haiku-4-5"
    assert "effort" not in req["output_config"]  # Haiku 4.5 400s on effort
    (row,) = await rows(db)
    assert row.cost_usd_micros == 150_000 * 1 + 900 * 5  # no long tier on 4.5


async def test_rollback_to_opus_5_keeps_effort_where_configured(api, url, db) -> None:
    api.script.append(
        message("claude-opus-5", {"moments": []}, input_tokens=1_000, output_tokens=100)
    )
    router = make_router(url, db, standard="claude-haiku-4-5", premium="claude-opus-5")
    with bound_tenant(TENANT):
        await router.complete(
            tenant_id=TENANT,
            purpose="viral_detection",
            system="s",
            user="u",
            schema=ViralMomentList,
            quality="premium",
        )
    req = api.requests[0]
    assert req["model"] == "claude-opus-5" and req["output_config"]["effort"] == "low"
    (row,) = await rows(db)
    assert row.cost_usd_micros == 1_000 * 5 + 100 * 25


# ---- regression: every routed schema is accepted as an output format ------


class _Plain(BaseModel):
    answer: str


@pytest.mark.parametrize(
    "schema",
    [
        HookBatch,
        ViralMomentList,
        RescoreVerdict,
        CropBoxVerdict,
        ThumbnailPickVerdict,
        FaceEmotionVerdict,
        _Plain,
    ],
)
def test_every_routed_schema_has_an_api_safe_format(schema: type[BaseModel]) -> None:
    from chalybclip.branding.logo import LogoSVG
    from chalybclip.llm.anthropic_provider import output_format_for

    for s in (schema, LogoSVG):
        fmt = output_format_for(s)
        text = json.dumps(fmt)
        for kw in ('"minimum"', '"maximum"', '"minLength"', '"maxLength"', '"exclusiveMinimum"'):
            assert kw not in text, (s.__name__, kw)
