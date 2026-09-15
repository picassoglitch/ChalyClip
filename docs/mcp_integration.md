# MCP integration — Claude Code, Cursor, Claude Desktop

The ChalyClip MCP server is a thin translation layer over the REST surface
(P3 #3). It runs locally over stdio so external agents drive the same
tenant a human would via the dashboard. This doc shows the config snippet
to register it with the three most common harnesses. No business logic is
exposed beyond what the dashboard already does.

## Prerequisites

```bash
# 1. Initialize SQLite + apply migrations.
chalybclip db init

# 2. Create a tenant + issue a full-scope API token.
chalybclip tenants add aldo "Aldo Villanueva"
chalybclip tokens issue --tenant aldo --scope full
# → tok_01H... (copy this; the CLI prints the raw token ONCE)
```

The token determines the tenant — every tool call binds that tenant via
contextvars. Rejected tokens fail fast at server boot.

## Claude Code

Add to `~/.claude.json` under your project:

```json
{
  "mcpServers": {
    "chalybclip": {
      "command": "C:/path/to/QuantorClipAI/.venv/Scripts/python.exe",
      "args": ["-m", "chalybclip.cli", "mcp", "serve"],
      "env": {
        "CHALYBCLIP_API_TOKEN": "tok_01H..."
      }
    }
  }
}
```

On macOS / Linux replace `Scripts/python.exe` with `bin/python`.

## Cursor

`~/.cursor/mcp.json` (same shape):

```json
{
  "mcpServers": {
    "chalybclip": {
      "command": "/path/to/QuantorClipAI/.venv/bin/python",
      "args": ["-m", "chalybclip.cli", "mcp", "serve"],
      "env": {
        "CHALYBCLIP_API_TOKEN": "tok_01H..."
      }
    }
  }
}
```

## Claude Desktop

`claude_desktop_config.json` (location varies by OS — see Anthropic docs).
Same structure as Claude Code.

## What the agent sees

After registration, ask the agent something like "List my recent streams
and show me which clips are still pending review" and it will call:

* `list_streams` — every stream for the bound tenant.
* `list_clips(stream_id)` — every clip for one stream.
* `get_clip(clip_id)` — clip detail + variants + the
  confidence-breakdown panel data + the list of allowed status
  transitions from the clip's current state.

For the publish loop:

* `update_clip_status(clip_id, "approved")` — moves the clip through the
  REST PATCH transition map (cut → ready_for_review → approved →
  published).
* `publish_clip(clip_id, variant_id, account_ids?)` — enqueues one
  `publish_jobs` row per connected account; the FastAPI lifespan loop
  drains them every 60s automatically.

For cost / measurement:

* `get_cost_projection` — today / MTD / projected EOM USD, plus
  per-purpose and per-model breakdowns.
* `get_calibration(platform)` — Pearson r over (rescore_score, views)
  for the last 30 days. Read this before deciding to enable any
  auto-publish toggles (currently still off everywhere).

State-transition tools all require `scope=full` on the API token.
Read-only tokens get a clear refusal and the agent stops trying.

## Troubleshooting

* **"no API token"** at boot: the `CHALYBCLIP_API_TOKEN` env var is empty
  in the agent's spawned subprocess. Most harnesses accept the `env`
  block above; otherwise pass `--token tok_...` in `args` directly.
* **"unknown token"**: the token wasn't issued for this DB. Check the
  `--db-path` flag if you're running from a non-default working
  directory; the CLI defaults to `./chalybclip.db`.
* **No tools listed**: agent didn't successfully spawn the subprocess.
  Run `chalybclip mcp serve --token tok_...` directly in a terminal to
  see the boot logs.
