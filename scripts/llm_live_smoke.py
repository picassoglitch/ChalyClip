"""Live smoke test: one tiny real call per routed purpose on the committed models.

    ANTHROPIC_API_KEY=sk-ant-... .venv/bin/python scripts/llm_live_smoke.py

Goes through the real LLMRouter + AnthropicProvider with config/llm.example.yaml,
prints model, tokens, cost and pass/fail per call, then the total. Stops issuing
calls once spend passes --max-usd (default $0.10); a full run costs a few cents.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import tempfile
import time
import zlib
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chalybclip.branding.logo import LogoSVG, sanitize_svg  # noqa: E402
from chalybclip.detect.viral import ViralMomentList  # noqa: E402
from chalybclip.llm import (  # noqa: E402
    CropBoxVerdict,
    FaceEmotionVerdict,
    LLMRouter,
    MultimodalImage,
    RescoreVerdict,
    ThumbnailPickVerdict,
)
from chalybclip.llm.config import load_llm_config  # noqa: E402
from chalybclip.variants.hooks import generate_hooks  # noqa: E402

TENANT = "ten_smoke"


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approve: bool
    reason: str = Field(min_length=1, max_length=120)


def tiny_png(rgb: tuple[int, int, int] = (230, 60, 40), size: int = 16) -> bytes:
    """A solid-colour PNG, built by hand so the script needs no imaging library."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    row = b"\x00" + bytes(rgb) * size
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(row * size))
        + chunk(b"IEND", b"")
    )


def build_cases(router: LLMRouter) -> list[tuple[str, Callable[[], Awaitable[Any]]]]:
    png = [MultimodalImage(media_type="image/png", data=tiny_png())]

    def vision(schema: type[BaseModel], ask: str) -> Callable[[], Awaitable[Any]]:
        return lambda: router.complete_multimodal(
            tenant_id=TENANT,
            purpose="vision_rescore",
            system="Rate the frame. Be brief.",
            user=ask,
            images=png,
            schema=schema,
        )

    def text(
        purpose: str, schema: type[BaseModel], user: str, **kw: Any
    ) -> Callable[[], Awaitable[Any]]:
        return lambda: router.complete(
            tenant_id=TENANT,
            purpose=purpose,
            system="Answer in one short line.",
            user=user,
            schema=schema,
            **kw,
        )

    async def logo() -> LogoSVG:
        out = await router.complete(
            tenant_id=TENANT,
            purpose="logo_generation",
            system="Return one minimal SVG (a single circle, under 200 chars) and a 3-word style hint.",
            user='Brand "Ñandú". Primary #e63c28, accent #ffffff.',
            schema=LogoSVG,
        )
        sanitize_svg(out.svg)  # raises when the SVG is unusable
        return out

    return [
        (
            "hook_generation",
            lambda: generate_hooks(
                tenant_id=TENANT,
                persona_voice="Streamer mexicano, directo y con humor.",
                persona_language="es",
                transcript_snippet="¡No manches, qué golazo! 🔥",
                n=1,
                router=router,
            ),
        ),
        (
            "viral_detection",
            text(
                "viral_detection",
                ViralMomentList,
                "[0.00-6.00] ¡No manches, qué golazo de último minuto! 🔥\n"
                "Return at most one moment.",
            ),
        ),
        ("vision_rescore/RescoreVerdict", vision(RescoreVerdict, "Score this frame 0-1.")),
        ("vision_rescore/CropBoxVerdict", vision(CropBoxVerdict, "Pick a 9:16 crop.")),
        ("vision_rescore/ThumbnailPickVerdict", vision(ThumbnailPickVerdict, "Pick frame 0.")),
        ("vision_rescore/FaceEmotionVerdict", vision(FaceEmotionVerdict, "Any face?")),
        ("clip_scoring", text("clip_scoring", RescoreVerdict, "Score 0.5, reason 'ok'.")),
        (
            "variant_generation_visual",
            text("variant_generation_visual", Decision, "Approve: yes. Reason: 'ok'."),
        ),
        ("agent_decision", text("agent_decision", Decision, "Approve: yes. Reason: 'ok'.")),
        ("logo_generation", logo),
    ]


def last_row(log: Path) -> dict[str, Any]:
    lines = log.read_text("utf-8").strip().splitlines()
    return json.loads(lines[-1]) if lines else {}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-usd", type=float, default=0.10, help="stop calling past this spend")
    ap.add_argument("--base-url", help="send to this Messages API URL instead (dry runs)")
    args = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    cfg = load_llm_config(ROOT / "config" / "llm.example.yaml")
    # One attempt with 512 output tokens: a single Opus call can't overshoot the $0.10 cap.
    anth_update: dict[str, object] = {"max_output_tokens": 512}
    if args.base_url:
        anth_update["base_url"] = args.base_url
    anth = cfg.providers["anthropic"].model_copy(update=anth_update)
    cfg = cfg.model_copy(
        update={
            "providers": {**cfg.providers, "anthropic": anth},
            "retry": cfg.retry.model_copy(update={"max_attempts": 1}),
        }
    )
    log = Path(tempfile.mkdtemp(prefix="llm-smoke-")) / "llm_calls.jsonl"
    router = LLMRouter(cfg, call_log_path=log)
    print(
        f"standard={cfg.model_for('anthropic', 'standard')} "
        f"premium={cfg.model_for('anthropic', 'premium')}\n"
    )
    print(f"{'purpose':38} {'model':18} {'in':>6} {'out':>6} {'att':>3} {'cost $':>9}  result")

    total_um, failures = 0, 0
    for name, call in build_cases(router):
        if total_um / 1e6 >= args.max_usd:
            print(f"{name:38} skipped: spend cap ${args.max_usd:.2f} reached")
            failures += 1
            continue
        t0 = time.monotonic()
        before = len(log.read_text("utf-8").splitlines()) if log.exists() else 0
        try:
            await call()
            result = "PASS"
        except Exception as e:
            result = f"FAIL {type(e).__name__}: {str(e)[:160]}"
            failures += 1
        row = (
            last_row(log)
            if log.exists() and len(log.read_text("utf-8").splitlines()) > before
            else {}
        )
        cost = int(row.get("cost_usd_micros", 0))
        total_um += cost
        print(
            f"{name:38} {row.get('model', '-'):18} {row.get('input_tokens', 0):>6} "
            f"{row.get('output_tokens', 0):>6} {row.get('attempts', 0):>3} {cost / 1e6:>9.5f}  "
            f"{result} ({time.monotonic() - t0:.1f}s)"
        )

    print(f"\ntotal cost ${total_um / 1e6:.5f}  ({failures} failed)  call log: {log}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
