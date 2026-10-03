"""Committed LLM price table — the source of `cost_usd_micros` in prod.

`config/llm.yaml` is gitignored and prod loads `config/llm.example.yaml`,
whose `pricing:` block only listed the two models routed today. Anything
else (a model swap, a dated snapshot id, a fallback) used to price at 0 —
and a 0-cost event is billed by the hub as 1 token. This table is code, so
it ships with every build and can't be shadowed by a local YAML: the YAML
`pricing:` block still wins per model (tests, deliberate overrides), this
fills every gap.

USD per 1M tokens, Anthropic first-party list prices (checked 2026-10-03
against Anthropic's model table). Cache pricing follows Anthropic's
multipliers on the input rate unless an entry overrides it: cache read =
0.1x, 5-minute cache write = 1.25x. Overrides: Opus 5.5 reads at $0.20
(0.05x), Fable 5.1 / Mythos 5.1 at $0.25 (0.025x).

Lookup matches the exact id first, then the longest family prefix, so a
dated snapshot (`claude-haiku-4-5-20251001`) prices as its family.
"""

from __future__ import annotations

from dataclasses import dataclass

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_5M_MULTIPLIER = 1.25


@dataclass(frozen=True)
class TokenRates:
    """USD per 1M tokens for one model."""

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None

    @property
    def cache_read_rate(self) -> float:
        return self.cache_read if self.cache_read is not None else self.input * CACHE_READ_MULTIPLIER

    @property
    def cache_write_rate(self) -> float:
        return (
            self.cache_write
            if self.cache_write is not None
            else self.input * CACHE_WRITE_5M_MULTIPLIER
        )


# Family prefixes → rates. Keep the most specific prefix for a family that
# changed price between point releases (Opus 4 / 4.1 vs 4.5+).
ANTHROPIC_RATES: dict[str, TokenRates] = {
    "claude-haiku-4-5": TokenRates(input=1.00, output=5.00),
    "claude-sonnet-4": TokenRates(input=3.00, output=15.00),  # 4, 4.5, 4.6
    "claude-sonnet-5": TokenRates(input=2.00, output=10.00),  # 5, 5.5
    "claude-opus-4-0": TokenRates(input=15.00, output=75.00),
    "claude-opus-4-1": TokenRates(input=15.00, output=75.00),
    "claude-opus-4-2": TokenRates(input=15.00, output=75.00),  # bare "claude-opus-4-2025…" snapshots
    "claude-opus-4-5": TokenRates(input=5.00, output=25.00),
    "claude-opus-4-6": TokenRates(input=5.00, output=25.00),
    "claude-opus-4-7": TokenRates(input=5.00, output=25.00),
    "claude-opus-4-8": TokenRates(input=5.00, output=25.00),
    "claude-opus-5": TokenRates(input=5.00, output=25.00),
    "claude-opus-5-5": TokenRates(input=4.00, output=20.00, cache_read=0.20),
    "claude-fable-5": TokenRates(input=10.00, output=50.00),  # 5: reads at 0.1x = $1
    "claude-fable-5-1": TokenRates(input=10.00, output=50.00, cache_read=0.25),
    "claude-mythos-5": TokenRates(input=10.00, output=50.00),
    "claude-mythos-5-1": TokenRates(input=10.00, output=50.00, cache_read=0.25),
}

# What an unknown model of a PAID provider is charged at: the most
# expensive rate in the table. Over-billing an unpriced model is a bug we
# notice and fix (an error is logged on every call); under-billing it at 0
# was silent.
CONSERVATIVE_RATES: TokenRates = max(
    ANTHROPIC_RATES.values(), key=lambda r: (r.output, r.input)
)

PROVIDER_RATES: dict[str, dict[str, TokenRates]] = {"anthropic": ANTHROPIC_RATES}


def lookup_rates(provider: str, model: str) -> TokenRates | None:
    """Exact id, then the longest matching family prefix. None if unknown."""
    table = PROVIDER_RATES.get(provider)
    if not table:
        return None
    if model in table:
        return table[model]
    best: str | None = None
    for prefix in table:
        if model.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    return table[best] if best is not None else None


def cost_micros(
    rates: TokenRates,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> int:
    """Real USD micros for one call. USD per 1M tokens == micros per token,
    so the rates multiply token counts directly."""
    cost = (
        max(0, input_tokens) * rates.input
        + max(0, output_tokens) * rates.output
        + max(0, cache_read_tokens) * rates.cache_read_rate
        + max(0, cache_write_tokens) * rates.cache_write_rate
    )
    return round(cost)
