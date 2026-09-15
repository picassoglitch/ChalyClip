"""Boot the ChalybClip dashboard locally with auto-drains on.

Convenience launcher so you can `python run.py` from the project root
instead of typing the uvicorn factory invocation. Reads `CHALYBCLIP_DB_PATH`
+ `CHALYBCLIP_HOST` + `CHALYBCLIP_PORT` from the environment with sensible
defaults.

Usage (after activating the venv):

    .venv\\Scripts\\activate            # PowerShell / cmd
    source .venv/Scripts/activate       # Git Bash
    source .venv/bin/activate           # macOS / Linux

    python run.py
    # → http://127.0.0.1:8000/dashboard/login
"""

from __future__ import annotations

import asyncio
import faulthandler
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import uvicorn

# Dump a Python traceback on hard crashes (segfaults, abort signals, CUDA
# OOMs). Without this, Whisper on long videos can SIGABRT mid-batch and the
# only signal the user sees is 'PS prompt comes back' — no error, no
# traceback, just a silently exited python.exe. faulthandler hooks SIGSEGV /
# SIGABRT / SIGBUS / SIGFPE / SIGILL so we get at least a partial stack.
faulthandler.enable()

from chalybclip.api import create_app
from chalybclip.db import Database, apply_migrations


def _load_dotenv() -> None:
    """Read `.env` into os.environ before anything else imports settings.

    Without this, ANTHROPIC_API_KEY never makes it into the process
    environment. The LLM router then caches the provider as None at init
    time and runs report the misleading "all providers failed for
    purpose=X: provider not available: anthropic" error — even though
    the user has a real Anthropic key in .env.

    We import dotenv lazily because it's optional: if the user has env vars
    set globally (CI, docker, manual export), we don't want to require the
    package.
    """
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        # `override=True` because Windows can leak empty-string values into
        # the environment from user/system profiles. Without override, an
        # ANTHROPIC_API_KEY="" set globally silently shadows the real value
        # in .env — and the LLM router then reports "provider not available"
        # for every Claude call. The .env file is the source of truth.
        load_dotenv(env_path, override=True)
    except ImportError:
        print(
            "[warn] python-dotenv not installed — .env file ignored. "
            "Run: pip install python-dotenv"
        )


def _looks_like_cookie_content(value: str) -> bool:
    """True if `value` is a cookies.txt BODY rather than a filesystem path.

    A real path is a single line with no tabs; an exported cookies.txt is
    multi-line, tab-delimited, and usually starts with a `# Netscape` header.
    """
    return (
        "\n" in value
        or "\t" in value
        or value.lstrip().startswith("# Netscape")
    )


def _materialize_cookies_file() -> None:
    """Write inline yt-dlp cookies (CHALYBCLIP_COOKIES_TXT) to a file and point
    CHALYBCLIP_COOKIES_FILE at it.

    On a datacenter IP (Cloud Run/Railway/Fly/etc.) YouTube blocks anonymous yt-dlp with
    "Sign in to confirm you're not a bot. Use --cookies-from-browser or
    --cookies." `cookies_from_browser` is useless on a headless host (no
    browser profile), and dropping a cookies.txt *file* onto a PaaS is awkward.
    So we let the operator paste their exported Netscape cookies.txt straight
    into a secret env var:

        CHALYBCLIP_COOKIES_TXT       — the cookies.txt content
        CHALYBCLIP_COOKIES_TXT_B64=1 — optional: the value is base64 (avoids
                                     newline/quoting mangling in the env var)

    At boot we materialize it to a real file and set CHALYBCLIP_COOKIES_FILE,
    which `Settings.cookies_file` reads — so every ingest path (channel poll,
    URL job, manual upload) picks it up via `ingest_vod`'s settings fallback,
    no per-call wiring needed.

    Forgiving by design: people routinely paste the cookies.txt *body* into
    CHALYBCLIP_COOKIES_FILE (the "file" var) by mistake — yt-dlp then tries to
    open a filename that is the entire cookie text and dies with a confusing
    FileNotFoundError (and leaks the cookies into the error log). So we accept
    the content in EITHER var and normalize it to a real file; only a genuine
    path in CHALYBCLIP_COOKIES_FILE (no newlines/tabs) is left untouched.
    """
    raw = os.environ.get("CHALYBCLIP_COOKIES_TXT")
    existing_file = os.environ.get("CHALYBCLIP_COOKIES_FILE")

    if raw:
        content = raw
        b64 = os.environ.get("CHALYBCLIP_COOKIES_TXT_B64", "").strip().lower()
        if b64 in ("1", "true", "yes"):
            import base64

            try:
                content = base64.b64decode(raw).decode("utf-8")
            except (ValueError, UnicodeDecodeError) as e:
                print(f"[cookies] CHALYBCLIP_COOKIES_TXT_B64 set but decode failed: {e}")
                return
    elif existing_file and _looks_like_cookie_content(existing_file):
        # The cookies.txt body was pasted into CHALYBCLIP_COOKIES_FILE — treat
        # it as content, not a path, and rewrite the var to a real file below.
        print(
            "[cookies] CHALYBCLIP_COOKIES_FILE looks like cookie CONTENT, not a "
            "path — normalizing it to a real file."
        )
        content = existing_file
    else:
        # CHALYBCLIP_COOKIES_FILE is a genuine path (or nothing is set) — leave
        # it for Settings.cookies_file to read as-is.
        return

    # Basic sanity: a Netscape cookies.txt is tab-delimited with a header line.
    # Don't hard-fail on a malformed file (yt-dlp will surface its own error),
    # but warn so a pasted-wrong value is obvious in the boot log.
    if "youtube.com" not in content and "google.com" not in content:
        print(
            "[cookies] cookie value has no youtube.com/google.com lines "
            "— is this a real exported cookies.txt? Writing it anyway."
        )

    import contextlib
    import tempfile

    cookies_path = Path(tempfile.gettempdir()) / "chalybclip-youtube-cookies.txt"
    try:
        cookies_path.write_text(content, encoding="utf-8")
    except OSError as e:
        print(f"[cookies] could not write cookies file: {e}")
        return
    # cookies are credentials — don't leave them world-readable.
    with contextlib.suppress(OSError):
        os.chmod(cookies_path, 0o600)
    os.environ["CHALYBCLIP_COOKIES_FILE"] = str(cookies_path.resolve())
    print(
        f"[cookies] yt-dlp cookies materialized to {cookies_path} "
        f"({len(content)} bytes) — YouTube ingest will authenticate with them."
    )


def _ensure_ffmpeg_on_path() -> None:
    """Best-effort: if ffmpeg isn't on PATH but winget installed it under
    %LOCALAPPDATA%\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg*, prepend that
    bin dir to PATH for this process.

    The winget ffmpeg packages install the binary under a versioned dir
    inside the package — they DON'T add it to the user PATH automatically,
    which leaves the user with `ffmpeg : not recognized` after a successful
    install. Fix that here so `python run.py` works on a fresh box without
    a PATH-editing step.
    """
    if shutil.which("ffmpeg"):
        return
    if os.name != "nt":
        return
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        return
    winget_packages = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
    if not winget_packages.exists():
        return
    # Both Gyan.FFmpeg and Gyan.FFmpeg.Essentials nest a versioned
    # ffmpeg-*-build/bin/ffmpeg.exe inside their package dir.
    for pkg in winget_packages.glob("Gyan.FFmpeg*"):
        for bin_dir in pkg.glob("ffmpeg-*/bin"):
            if (bin_dir / "ffmpeg.exe").exists():
                os.environ["PATH"] = (
                    f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                )
                print(f"Located ffmpeg at {bin_dir} (added to PATH for this run)")
                return


def _verify_or_fallback_to_cpu() -> None:
    """If `CHALYBCLIP_WHISPER_DEVICE=cuda` (the default) but cuBLAS isn't actually
    loadable, override to CPU for this process.

    Without this, the dashboard happily accepts uploads, runs analyze_video,
    and only blows up at the transcribe step with a cryptic
    'Library cublas64_12.dll is not found' — burning 30+ seconds of user time
    per attempt. We try to load cublas64_12.dll up front; if it fails, we
    print a loud warning and switch to CPU + int8 + base model so the rest
    of the pipeline still works.
    """
    if os.name != "nt":
        return
    device = os.environ.get("CHALYBCLIP_WHISPER_DEVICE", "cuda").strip().lower()
    if device != "cuda":
        return  # User already opted into cpu — nothing to verify.
    import ctypes

    try:
        ctypes.WinDLL("cublas64_12.dll")
        return  # Loaded fine — CUDA path is genuinely available.
    except OSError:
        pass

    # CUDA 13.x installs ship cublas64_13.dll, not cublas64_12.dll —
    # CTranslate2 / faster-whisper hard-code the v12 filename. If we see
    # nvcc on PATH but cuBLAS v12 still fails, surface that specific mismatch.
    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path and "v13" in cuda_path.lower():
        print(
            f"\n[CUDA fallback] CUDA 13.x is installed at {cuda_path} but "
            f"faster-whisper / CTranslate2 needs the v12 cuBLAS library "
            f"(cublas64_12.dll). Fix: 'pip install nvidia-cublas-cu12 "
            f"nvidia-cudnn-cu12' inside the venv — the pip packages provide "
            f"the v12 DLLs alongside your existing v13 install. Falling back "
            f"to CPU for this run.\n"
        )
    else:
        print(
            "\n[CUDA fallback] cublas64_12.dll could not be loaded — falling "
            "back to CPU transcription for this run. Install the full CUDA "
            "Toolkit from NVIDIA if you want GPU speed; for now, CPU + base "
            "+ int8 will work.\n"
        )
    os.environ["CHALYBCLIP_WHISPER_DEVICE"] = "cpu"
    os.environ["CHALYBCLIP_WHISPER_COMPUTE_TYPE"] = "int8"
    # Only downgrade the model size if the user hasn't picked one explicitly.
    # `medium` on CPU is bearable but slow; `base` is the better default.
    if not os.environ.get("CHALYBCLIP_WHISPER_MODEL"):
        os.environ["CHALYBCLIP_WHISPER_MODEL"] = "base"
    # Bust the settings cache so the override takes effect.
    try:
        from chalybclip.settings import get_settings

        get_settings.cache_clear()
    except Exception:
        pass


def _ensure_system_cuda_on_path() -> None:
    """Add the NVIDIA CUDA Toolkit `bin\\` dir to PATH if installed system-wide.

    The Toolkit installer adds its bin dir to the *system* PATH at install
    time, but already-open shells don't pick up the change. Result: the
    user installs CUDA, restarts python run.py from the same shell, and
    still hits 'cublas64_12.dll not found' until they reopen PowerShell.

    Walk `C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*` and add
    every versioned bin dir we find. No-op on Linux / macOS.
    """
    if os.name != "nt":
        return
    roots = [
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "NVIDIA GPU Computing Toolkit"
        / "CUDA",
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "NVIDIA GPU Computing Toolkit"
        / "CUDA",
    ]
    # CUDA_PATH is the canonical env var the installer sets when it succeeds.
    cuda_path_env = os.environ.get("CUDA_PATH")
    if cuda_path_env:
        roots.insert(0, Path(cuda_path_env).parent)

    added: list[str] = []
    for cuda_root in roots:
        if not cuda_root.exists():
            continue
        for version_dir in sorted(cuda_root.iterdir(), reverse=True):
            bin_dir = version_dir / "bin"
            if not bin_dir.is_dir():
                continue
            # Only add if cuBLAS is actually there (filters out half-installed
            # versions / non-bin subdirs).
            if not any(bin_dir.glob("cublas64_*.dll")):
                continue
            try:
                os.add_dll_directory(str(bin_dir))
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            added.append(f"{cuda_root.name}/{version_dir.name}")
    if added:
        print(
            f"Registered system CUDA Toolkit bin dirs for the DLL loader: "
            f"{', '.join(added)}"
        )


def _ensure_cuda_libs_on_path() -> None:
    """Best-effort: add bundled nvidia-* pip packages' DLL dirs to the search path.

    faster-whisper / CTranslate2 dynamically link cublas64_12.dll and the
    cuDNN libraries. The CUDA Toolkit installer adds them to PATH, but most
    users just `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` — which
    drops the DLLs at .venv/Lib/site-packages/nvidia/*/bin/* but DOESN'T
    register them with Windows' DLL loader. Result: a confusing
    'Library cublas64_12.dll is not found or cannot be loaded' the first
    time transcribe runs, even though `pip list` shows the package.

    Walk site-packages/nvidia/<lib>/bin once at startup and register each
    via os.add_dll_directory + PATH so faster-whisper's load-time link
    resolves cleanly. No-op on Linux / macOS — they use rpath / DYLD.
    """
    if os.name != "nt":
        return
    try:
        import sysconfig

        site_packages = Path(sysconfig.get_paths()["purelib"])
    except Exception:
        return
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.exists():
        # Helpful breadcrumb when the user thinks they pip-installed but
        # didn't, or installed into a different interpreter.
        print(
            f"No nvidia/* CUDA pip packages found under {site_packages}. "
            f"If you intended GPU transcription, set CHALYBCLIP_WHISPER_DEVICE=cpu "
            f"in .env (CPU works without CUDA libs) or install the full "
            f"CUDA Toolkit from NVIDIA's website."
        )
        return
    added: list[str] = []
    for lib_dir in nvidia_root.iterdir():
        bin_dir = lib_dir / "bin"
        if not bin_dir.is_dir():
            continue
        try:
            os.add_dll_directory(str(bin_dir))
        except (OSError, AttributeError):
            pass
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
        added.append(lib_dir.name)
    if added:
        print(
            f"Registered CUDA pip libs for the DLL loader: {', '.join(sorted(added))}"
        )


def _print_whisper_config() -> None:
    """Echo the resolved Whisper settings at boot.

    Helps the user catch a stale .env (or one that's not being loaded) before
    the first upload. If they see `device: cuda` but expected `cpu`, the
    settings file isn't where pydantic-settings is looking.
    """
    try:
        from chalybclip.settings import get_settings

        s = get_settings()
        print(
            f"Whisper config: device={s.whisper_device} model={s.whisper_model} "
            f"compute={s.whisper_compute_type}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"[warn] couldn't read settings: {e}")


def _resolve_python_check() -> None:
    """Hard-stop if you launched via system Python 3.14 instead of the venv.

    faster-whisper requires Python <= 3.12; running under 3.14 will fail
    with cryptic CTranslate2 errors deep in the transcription path.
    """
    if sys.version_info >= (3, 13):
        print(
            f"ERROR: ChalybClip requires Python 3.11 or 3.12, got {sys.version_info.major}."
            f"{sys.version_info.minor}.\n"
            f"Activate the venv first:\n"
            f"  PowerShell:  .venv\\Scripts\\Activate.ps1\n"
            f"  Git Bash:    source .venv/Scripts/activate\n"
            f"  macOS/Linux: source .venv/bin/activate",
            file=sys.stderr,
        )
        sys.exit(1)


def _silence_connection_reset(loop: asyncio.AbstractEventLoop) -> None:
    """Filter the Windows-only 'WinError 10054' spam out of the asyncio log.

    The dashboard serves the source video and clip MP4s as streaming
    FileResponses. Every time the browser closes a Range request mid-flight
    (user seeks the player, navigates away, etc.) the Proactor event loop's
    transport shutdown trips `ConnectionResetError(WinError 10054)`, and
    asyncio's default handler logs a full multi-line traceback for each
    one. They're harmless and pollute the terminal — a single page-load
    can produce 5+ of them.

    Install a custom exception handler that swallows just this specific
    case and lets every other unhandled exception through to the default.
    """
    if os.name != "nt":
        return
    prev = loop.get_exception_handler()

    def handler(loop_: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        winerror = getattr(exc, "winerror", None)
        is_reset = isinstance(exc, ConnectionResetError) and winerror == 10054
        # Some failure modes don't expose .winerror — pattern-match the
        # message as a fallback.
        msg = context.get("message", "")
        is_reset_msg = "10054" in str(exc) or "10054" in str(msg)
        if is_reset or is_reset_msg:
            return
        if prev is not None:
            prev(loop_, context)
        else:
            loop_.default_exception_handler(context)

    loop.set_exception_handler(handler)


def _free_stale_port(host: str, port: int) -> None:
    """If `host:port` is already bound, try to kill the owning process.

    Otherwise uvicorn fails with WinError 10048 ('only one usage of each
    socket address ... is normally permitted') and exits — leaving the
    user to manually hunt for the stale python.exe. Almost always the
    stale process is a previous `python run.py` that the user Ctrl+C'd
    in a way that orphaned it (closed terminal, OS suspend, etc.) so
    auto-cleanup is the right default.

    Only operates on 127.0.0.1 / localhost. If host is something else
    (LAN bind, container), bail — we don't want to kill a real shared
    service by accident.
    """
    import socket

    if host not in ("127.0.0.1", "localhost", "::1"):
        return

    # Cheap probe — try to bind ourselves; if we succeed, port is free.
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    try:
        probe.bind((host, port))
    except OSError:
        pass
    else:
        probe.close()
        return
    finally:
        try:
            probe.close()
        except OSError:
            pass

    print(f"[port-cleanup] {host}:{port} is in use — trying to free it.")

    if os.name != "nt":
        print(
            f"[port-cleanup] Auto-cleanup is Windows-only; on Linux/macOS run: "
            f"  lsof -t -i:{port} | xargs kill -9"
        )
        return

    # Resolve PID(s) holding the port via PowerShell. Get-NetTCPConnection
    # is the cleanest API; netstat -ano is the universal fallback.
    import subprocess

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"(Get-NetTCPConnection -LocalPort {port} -State Listen "
                f"-ErrorAction SilentlyContinue).OwningProcess",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        pids = {
            int(line) for line in result.stdout.split() if line.strip().isdigit()
        }
    except Exception as e:  # noqa: BLE001
        print(f"[port-cleanup] could not enumerate listeners: {e}")
        return

    if not pids:
        print(f"[port-cleanup] No listener PID found — port may free up on its own.")
        return

    me = os.getpid()
    for pid in pids:
        if pid == me:
            continue
        try:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            print(f"[port-cleanup] killed stale process pid={pid}")
        except Exception as e:  # noqa: BLE001
            print(f"[port-cleanup] failed to kill pid={pid}: {e}")

    # Brief breather so Windows fully releases the socket before we bind.
    import time as _time

    _time.sleep(0.5)


def _check_db_persistence(db_path: Path) -> None:
    """Loud warning when the DB file lives on container-ephemeral storage.

    Slice O.42 — Cloud Run / Railway / Fly.io container filesystems are wiped
    on every redeploy. CHALYBCLIP_DB_PATH defaults to /data/chalybclip.db in
    the Dockerfile, which is meaningless unless a persistent Volume is
    actually mounted at /data on the dashboard side. Operators forget
    that step and then lose all their tenants on the next deploy.

    Detection: on Linux, a real volume mount has a different st_dev than
    the root filesystem. Same device = the directory is just part of the
    container's overlay = ephemeral. We log a loud warning rather than
    refusing to boot so local dev (where the file lives in CWD) still
    works without ceremony.
    """
    db_dir = db_path.parent.resolve()
    db_dir.mkdir(parents=True, exist_ok=True)
    # Only the "/data on container" case is risky; local paths are
    # expected to be ephemeral relative to the dev's intent.
    if not str(db_dir).startswith(("/data", "/mnt", "/var/lib")):
        return
    try:
        if os.stat(db_dir).st_dev == os.stat("/").st_dev:
            banner = (
                "\n"
                + "=" * 70
                + "\n"
                + "  WARNING: CHALYBCLIP_DB_PATH points at " + str(db_path) + "\n"
                + "  but " + str(db_dir) + " is NOT a persistent mount.\n"
                + "  \n"
                + "  Every redeploy will wipe this DB. Attach a persistent\n"
                + "  Volume (Cloud Run volume / Railway volume -> mount /data)\n"
                + "  before going live, or set CHALYBCLIP_DB_PATH to a path\n"
                + "  on a volume you control.\n"
                + "=" * 70
                + "\n"
            )
            print(banner, file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 — never block boot on the check
        pass


async def _boot() -> None:
    _silence_connection_reset(asyncio.get_running_loop())
    # Postgres takes precedence when DATABASE_URL is set (injected by Cloud Run from Secret Manager);
    # otherwise fall back to the local SQLite file. The SQLite-only
    # ephemeral-storage warning is meaningless for a managed Postgres, so
    # skip it on the DSN path.
    database_url = os.environ.get("DATABASE_URL")
    db_path = Path(os.environ.get("CHALYBCLIP_DB_PATH", "./chalybclip.db"))
    host = os.environ.get("CHALYBCLIP_HOST", "127.0.0.1")
    port = int(os.environ.get("CHALYBCLIP_PORT", "8000"))

    _free_stale_port(host, port)
    if database_url:
        db = Database(database_url)
    else:
        _check_db_persistence(db_path)
        db = Database(db_path)
    await apply_migrations(db)
    app = create_app(db=db, enable_background_drains=True)

    print(f"\nChalybClip dashboard: http://{host}:{port}/dashboard/login\n")
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


def main() -> None:
    _resolve_python_check()
    _load_dotenv()
    _materialize_cookies_file()
    _ensure_ffmpeg_on_path()
    _ensure_system_cuda_on_path()
    _ensure_cuda_libs_on_path()
    _verify_or_fallback_to_cpu()
    _print_whisper_config()
    try:
        asyncio.run(_boot())
    except KeyboardInterrupt:
        # Ctrl+C is the documented way to stop the dev server; exit
        # cleanly so the user doesn't see a scary-looking traceback.
        print("\nChalybClip dashboard stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
