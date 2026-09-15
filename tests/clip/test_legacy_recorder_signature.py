"""Render Migration R7 — pin the legacy seek-and-shoot's progress_callback
plumbing.

Operator hit a NameError in production: `_record_via_seek_and_shoot` at
preview_recorder.py:450 referenced `progress_callback` but the function
signature had never received it. T1 added `progress_callback` to
`record_clip_to_mp4` but the helper signature was never updated, so the
call from the parent silently dropped it AND the helper body referenced
an undefined name at runtime.

This is the cheap test that catches the next time someone adds a new
log field / progress hook that references a name the helper doesn't
own. AST-level — no Playwright, no subprocess, no I/O.
"""
from __future__ import annotations

import ast
import builtins
from pathlib import Path


_RECORDER_PATH = (
    Path(__file__).resolve().parents[2]
    / "chalybclip" / "clip" / "preview_recorder.py"
)


def _function(name: str) -> ast.AsyncFunctionDef:
    tree = ast.parse(_RECORDER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AsyncFunctionDef)
            and node.name == name
        ):
            return node
    raise AssertionError(f"no async function named {name!r}")


def _free_names(fn: ast.AsyncFunctionDef) -> set[str]:
    """Names loaded in fn that are NOT parameters or defined locally
    inside fn (and not Python builtins). These need to be supplied by
    the module scope — if any of them are typos / dropped parameters,
    they'll surface as NameError at runtime."""
    params: set[str] = {a.arg for a in fn.args.args}
    params |= {a.arg for a in fn.args.kwonlyargs}
    if fn.args.vararg:
        params.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        params.add(fn.args.kwarg.arg)

    bound: set[str] = set(params)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, ast.For):
            for tgt in ast.walk(node.target):
                if isinstance(tgt, ast.Name):
                    bound.add(tgt.id)
        elif isinstance(node, ast.withitem) and node.optional_vars:
            for tgt in ast.walk(node.optional_vars):
                if isinstance(tgt, ast.Name):
                    bound.add(tgt.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)

    loaded: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)

    return {n for n in loaded - bound if n not in dir(builtins)}


def test_record_via_seek_and_shoot_takes_progress_callback() -> None:
    """The helper must accept progress_callback as a kw-only argument.
    Regression-pinning the exact production crash."""
    fn = _function("_record_via_seek_and_shoot")
    params = {a.arg for a in fn.args.kwonlyargs}
    assert "progress_callback" in params, (
        "_record_via_seek_and_shoot must accept progress_callback — "
        "operator hit NameError in production"
    )


def test_record_via_seek_and_shoot_takes_clip_id() -> None:
    """clip_id is referenced by the periodic progress log (and the
    progress_callback warning); without it being on the signature,
    referencing it inside the loop raises NameError."""
    fn = _function("_record_via_seek_and_shoot")
    params = {a.arg for a in fn.args.kwonlyargs}
    assert "clip_id" in params, (
        "_record_via_seek_and_shoot needs clip_id for telemetry/logs"
    )


def test_record_via_seek_and_shoot_has_no_unresolved_names() -> None:
    """Belt-and-braces: every Name loaded inside the helper must be
    either a parameter, locally bound, a builtin, or a known
    module-level name imported/defined in preview_recorder.py.
    Catches any FUTURE T1-style mistake where a new field is added
    that references a dropped parameter."""
    fn = _function("_record_via_seek_and_shoot")
    unresolved = _free_names(fn)

    # Known module-level names that the helper legitimately reads:
    # imports + module-level constants + the parent module's other
    # functions. List is small + concrete so a typo at module level
    # also surfaces here.
    module_globals = {
        "Path", "asyncio", "tempfile",
        "PreviewRecordingError", "SCREENCAST_JPEG_QUALITY",
        "_log", "_encode_image_sequence",
    }
    leftover = unresolved - module_globals
    assert not leftover, (
        f"_record_via_seek_and_shoot references undefined names: "
        f"{sorted(leftover)}"
    )
