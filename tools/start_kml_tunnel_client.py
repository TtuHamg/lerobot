#!/usr/bin/env python3
"""Start the KML WebSocket tunnel client using cookies from a local browser.

The chosen browser must already have a valid KML SSO session:

    python tools/start_kml_tunnel_client.py                 # auto-detect a browser
    python tools/start_kml_tunnel_client.py --browser chrome
    python tools/start_kml_tunnel_client.py --browser edge
    python tools/start_kml_tunnel_client.py --browser firefox

The script does not print the cookie. It reads only cookies applicable to the
KML gateway URL from the selected browser, sets KML_COOKIE in the child process
environment, and execs tools/ws_tcp_tunnel.py client.

Firefox stores cookies in plaintext in ``cookies.sqlite``. Chromium-family
browsers (Chrome, Chromium, Edge, Brave) store them AES-encrypted in ``Cookies``
and need the per-browser key from the OS keyring (Secret Service); those code
paths require the ``secretstorage`` and ``pycryptodome`` packages, imported
lazily only when a Chromium browser is selected.
"""

from __future__ import annotations

import argparse
import configparser
import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit


DEFAULT_TARGET_HOST = "kml-dtmachine-27353-prod-0.kmlhb2az1l3-2.corp.kuaishou.com"
DEFAULT_WS_URL = f"wss://{DEFAULT_TARGET_HOST}/ws"
DEFAULT_TUNNEL_SCRIPT = Path(__file__).with_name("ws_tcp_tunnel.py")
DEFAULT_LEROBOT_PYTHON = Path("/home/pnp/miniconda3/envs/lerobot/bin/python")

FIREFOX_ROOTS = (
    Path.home() / "snap/firefox/common/.mozilla/firefox",
    Path.home() / ".mozilla/firefox",
)

# Order in which "auto" tries browsers.
AUTO_BROWSER_ORDER = ("chrome", "edge", "chromium", "brave", "firefox")


class ChromiumBrowser(NamedTuple):
    """Where a Chromium-family browser keeps its data and keyring password."""

    config_roots: tuple[Path, ...]
    # Preferred keyring "application" attribute values, most specific first.
    keyring_apps: tuple[str, ...]


CHROMIUM_BROWSERS: dict[str, ChromiumBrowser] = {
    "chrome": ChromiumBrowser(
        config_roots=(
            Path.home() / ".config/google-chrome",
            Path.home() / ".var/app/com.google.Chrome/config/google-chrome",
        ),
        keyring_apps=("chrome",),
    ),
    "chromium": ChromiumBrowser(
        config_roots=(
            Path.home() / ".config/chromium",
            Path.home() / "snap/chromium/common/chromium",
            Path.home() / ".var/app/org.chromium.Chromium/config/chromium",
        ),
        keyring_apps=("chromium",),
    ),
    "edge": ChromiumBrowser(
        config_roots=(
            Path.home() / ".config/microsoft-edge",
            Path.home() / ".config/microsoft-edge-beta",
            Path.home() / ".config/microsoft-edge-dev",
        ),
        # Edge normally uses "microsoft-edge"; some installs reuse the
        # "chromium" keyring entry, so keep it as a fallback candidate.
        keyring_apps=("microsoft-edge", "chromium"),
    ),
    "brave": ChromiumBrowser(
        config_roots=(
            Path.home() / ".config/BraveSoftware/Brave-Browser",
            Path.home() / ".var/app/com.brave.Browser/config/BraveSoftware/Brave-Browser",
        ),
        keyring_apps=("brave",),
    ),
}


class CookieRow(NamedTuple):
    """A browser-agnostic cookie row, with expiry already normalized to epoch seconds."""

    host: str
    name: str
    value: str
    path: str
    expiry_seconds: float  # 0 => session cookie (no expiry)
    is_secure: bool
    creation_order: int  # relative ordering key within one browser


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _domain_matches(cookie_host: str, target_host: str) -> bool:
    cookie_host = cookie_host.lower()
    target_host = target_host.lower()
    if cookie_host.startswith("."):
        domain = cookie_host[1:]
        return target_host == domain or target_host.endswith(f".{domain}")
    return target_host == cookie_host


def _path_matches(cookie_path: str, request_path: str) -> bool:
    if cookie_path == request_path:
        return True
    return request_path.startswith(cookie_path) and (
        cookie_path.endswith("/") or request_path[len(cookie_path) :].startswith("/")
    )


def _candidate_hosts(target_host: str) -> list[str]:
    labels = target_host.split(".")
    return [target_host, *(f".{'.'.join(labels[index:])}" for index in range(len(labels) - 1))]


def _file_state(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _open_sqlite_snapshot(db_path: Path, extra_suffixes: tuple[str, ...] = ("-wal",)):
    """Return a read-only sqlite connection, falling back to a consistent file copy.

    Live browsers may hold an exclusive lock (notably Snap builds). When the
    read-only URI open fails with a lock/busy error, snapshot the DB plus its
    sidecar files, rejecting any copy taken while the source was changing.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Cookie DB not found: {db_path}")

    live_uri = f"{db_path.resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(live_uri, uri=True, timeout=2.0)
        conn.execute("pragma query_only=on")
        conn.execute("begin")
        return conn
    except sqlite3.OperationalError as exc:
        if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
            raise

    sidecars = [Path(f"{db_path}{suffix}") for suffix in extra_suffixes]
    tmp_dir = Path(tempfile.mkdtemp(prefix="kml_cookies_"))
    for attempt in range(5):
        before = tuple(_file_state(p) for p in (db_path, *sidecars))
        attempt_dir = tmp_dir / str(attempt)
        attempt_dir.mkdir()
        snapshot = attempt_dir / "Cookies"
        try:
            shutil.copy2(db_path, snapshot)
            for sidecar in sidecars:
                if sidecar.exists():
                    shutil.copy2(sidecar, attempt_dir / sidecar.name)
        except FileNotFoundError:
            continue
        if before != tuple(_file_state(p) for p in (db_path, *sidecars)):
            continue
        try:
            conn = sqlite3.connect(snapshot)
            conn.execute("pragma query_only=on")
            return conn
        except sqlite3.DatabaseError:
            continue

    raise RuntimeError(f"Could not obtain a consistent snapshot of {db_path}")


# --------------------------------------------------------------------------- #
# Firefox
# --------------------------------------------------------------------------- #
def find_firefox_cookie_db() -> Path:
    for firefox_root in FIREFOX_ROOTS:
        profiles_ini = firefox_root / "profiles.ini"
        if not profiles_ini.exists():
            continue

        config = configparser.ConfigParser(interpolation=None)
        config.read(profiles_ini)
        profile_paths: list[tuple[str, bool]] = []
        for section in config.sections():
            if section.startswith("Install"):
                install_default = config.get(section, "Default", fallback="")
                if install_default:
                    profile_paths.append((install_default, True))

        profile_sections = [section for section in config.sections() if section.startswith("Profile")]
        profile_sections.sort(key=lambda section: config.get(section, "Default", fallback="0") != "1")
        for section in profile_sections:
            profile_path_value = config.get(section, "Path", fallback="")
            if not profile_path_value:
                continue
            profile_paths.append(
                (profile_path_value, config.getboolean(section, "IsRelative", fallback=True))
            )

        seen: set[Path] = set()
        for profile_path_value, is_relative in profile_paths:
            profile_path = Path(profile_path_value).expanduser()
            if is_relative:
                profile_path = firefox_root / profile_path
            if profile_path in seen:
                continue
            seen.add(profile_path)
            cookie_db = profile_path.expanduser() / "cookies.sqlite"
            if cookie_db.exists():
                return cookie_db

    raise FileNotFoundError(
        "Firefox cookies.sqlite was not found. Start Firefox and open the KML page at least once."
    )


def read_firefox_rows(cookie_db: Path, target_host: str) -> list[CookieRow]:
    conn = _open_sqlite_snapshot(cookie_db)
    try:
        schema_version = int(conn.execute("pragma user_version").fetchone()[0])
        columns = {row[1] for row in conn.execute("pragma table_info(moz_cookies)")}
        partition_filter = "and isPartitionedAttributeSet = 0" if "isPartitionedAttributeSet" in columns else ""
        candidate_hosts = _candidate_hosts(target_host)
        placeholders = ",".join("?" for _ in candidate_hosts)
        raw = conn.execute(
            f"""
            select host, name, value, path, expiry, isSecure, creationTime
            from moz_cookies
            where originAttributes = '' {partition_filter}
              and host in ({placeholders})
            order by length(path) desc, creationTime asc
            """,
            candidate_hosts,
        ).fetchall()
    finally:
        conn.close()

    rows: list[CookieRow] = []
    for host, name, value, path, expiry, is_secure, creation_time in raw:
        if value is None:
            continue
        expiry_seconds = expiry / 1000 if schema_version >= 16 else expiry
        rows.append(
            CookieRow(host, name, value, path, float(expiry_seconds or 0), bool(is_secure), int(creation_time or 0))
        )
    return rows


# --------------------------------------------------------------------------- #
# Chromium family (Chrome / Chromium / Edge / Brave)
# --------------------------------------------------------------------------- #
def find_chromium_cookie_db(browser: str, profile: str) -> Path:
    spec = CHROMIUM_BROWSERS[browser]
    for root in spec.config_roots:
        for candidate in (root / profile / "Cookies", root / profile / "Network" / "Cookies"):
            if candidate.exists():
                return candidate
    searched = ", ".join(str(root) for root in spec.config_roots)
    raise FileNotFoundError(
        f"{browser} cookie DB not found (looked under: {searched}). "
        f"Open the KML page in {browser} and complete SSO login at least once."
    )


def _keyring_passwords(preferred_apps: tuple[str, ...]) -> list[bytes]:
    """Collect candidate encryption passwords from the Secret Service keyring.

    Returns the preferred-app secrets first (most likely to be correct), then
    every other ``chrome_libsecret_os_crypt_password_v2`` secret as fallbacks,
    then the hardcoded ``peanuts`` used when no keyring is available.
    """
    passwords: list[bytes] = []
    try:
        import secretstorage
    except ImportError:
        return [b"peanuts"]

    try:
        conn = secretstorage.dbus_init()
    except Exception:
        return [b"peanuts"]

    def _secret(item) -> bytes | None:
        try:
            if item.is_locked():
                item.unlock()
            return item.get_secret()
        except Exception:
            return None

    seen: set[bytes] = set()

    def _add(secret: bytes | None) -> None:
        if secret and secret not in seen:
            seen.add(secret)
            passwords.append(secret)

    for app in preferred_apps:
        for item in secretstorage.search_items(conn, {"application": app}):
            _add(_secret(item))

    for item in secretstorage.search_items(
        conn, {"xdg:schema": "chrome_libsecret_os_crypt_password_v2"}
    ):
        _add(_secret(item))

    passwords.append(b"peanuts")
    return passwords


def _decrypt_chromium_value(encrypted: bytes, key: bytes, strip_hash_prefix: bool) -> str | None:
    """Decrypt one Chromium cookie value; return None if this key does not fit."""
    if not encrypted:
        return ""
    if encrypted[:3] not in (b"v10", b"v11"):
        # Unencrypted legacy value.
        try:
            return encrypted.decode("utf-8")
        except UnicodeDecodeError:
            return None

    from Crypto.Cipher import AES

    ciphertext = encrypted[3:]
    if len(ciphertext) < 16 or len(ciphertext) % 16 != 0:
        return None
    try:
        decrypted = AES.new(key, AES.MODE_CBC, b" " * 16).decrypt(ciphertext)
    except ValueError:
        return None
    pad = decrypted[-1]
    if pad < 1 or pad > 16 or pad > len(decrypted):
        return None
    decrypted = decrypted[:-pad]
    if strip_hash_prefix:
        # Chrome >= v24 prepends a 32-byte SHA-256 of the domain to the plaintext.
        decrypted = decrypted[32:]
    try:
        return decrypted.decode("utf-8")
    except UnicodeDecodeError:
        return None


# Chromium stores time as microseconds since 1601-01-01 UTC.
_CHROMIUM_EPOCH_OFFSET_S = 11644473600


def read_chromium_rows(cookie_db: Path, target_host: str, preferred_apps: tuple[str, ...]) -> list[CookieRow]:
    conn = _open_sqlite_snapshot(cookie_db, extra_suffixes=("-wal", "-shm"))
    try:
        meta_version = int(conn.execute("select value from meta where key='version'").fetchone()[0])
        candidate_hosts = _candidate_hosts(target_host)
        placeholders = ",".join("?" for _ in candidate_hosts)
        raw = conn.execute(
            f"""
            select host_key, name, encrypted_value, path, expires_utc, is_secure, creation_utc
            from cookies
            where host_key in ({placeholders})
            order by length(path) desc, creation_utc asc
            """,
            candidate_hosts,
        ).fetchall()
    finally:
        conn.close()

    if not raw:
        return []

    strip_hash_prefix = meta_version >= 24
    passwords = _keyring_passwords(preferred_apps)

    # Pick the password that decrypts the most rows; the correct key decrypts
    # every value, wrong keys fail PKCS7/UTF-8 on almost all of them.
    best_decrypted: list[str | None] | None = None
    best_score = -1
    for password in passwords:
        key = hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1, 16)
        decrypted = [_decrypt_chromium_value(enc, key, strip_hash_prefix) for _, _, enc, *_ in raw]
        score = sum(1 for value in decrypted if value is not None)
        if score > best_score:
            best_score, best_decrypted = score, decrypted
        if score == len(raw):
            break

    if not best_decrypted or best_score <= 0:
        raise RuntimeError(
            "Could not decrypt any Chromium cookies. The browser's keyring password was "
            "unavailable or did not match (is the login keyring unlocked?)."
        )

    rows: list[CookieRow] = []
    for (host, name, _enc, path, expires_utc, is_secure, creation_utc), value in zip(raw, best_decrypted):
        if value is None:
            continue
        expiry_seconds = (expires_utc / 1_000_000 - _CHROMIUM_EPOCH_OFFSET_S) if expires_utc else 0
        rows.append(
            CookieRow(host, name, value, path, float(expiry_seconds), bool(is_secure), int(creation_utc or 0))
        )
    return rows


# --------------------------------------------------------------------------- #
# Header assembly (browser-agnostic)
# --------------------------------------------------------------------------- #
def build_cookie_header(rows: list[CookieRow], target_host: str, request_path: str, *, secure: bool) -> str:
    now = int(time.time())
    cookie_parts: list[str] = []
    # Cookie-header order: longer paths first, then earlier creation time.
    for row in sorted(rows, key=lambda r: (-len(r.path), r.creation_order)):
        if row.value is None or not _domain_matches(row.host, target_host) or not _path_matches(row.path, request_path):
            continue
        if row.expiry_seconds and row.expiry_seconds <= now:
            continue
        if row.is_secure and not secure:
            continue
        if not row.name or any(
            character in ";=" or ord(character) < 0x20 or ord(character) == 0x7F for character in row.name
        ):
            continue
        if any(character == ";" or ord(character) < 0x20 or ord(character) == 0x7F for character in row.value):
            continue
        cookie_parts.append(f"{row.name}={row.value}")

    if not cookie_parts:
        raise RuntimeError(
            "No usable KML cookies were found. Open the KML page in the selected browser, "
            "complete SSO login, then run this script again."
        )

    return "; ".join(cookie_parts)


def collect_cookie_rows(browser: str, target_host: str, cookie_db_override: Path | None, profile: str) -> list[CookieRow]:
    if browser == "firefox":
        cookie_db = cookie_db_override or find_firefox_cookie_db()
        return read_firefox_rows(cookie_db, target_host)

    spec = CHROMIUM_BROWSERS[browser]
    cookie_db = cookie_db_override or find_chromium_cookie_db(browser, profile)
    return read_chromium_rows(cookie_db, target_host, spec.keyring_apps)


def resolve_browser(browser: str, target_host: str, request_path: str, secure: bool, profile: str) -> tuple[str, str]:
    """Return (browser, cookie_header). For 'auto', try each browser in order."""
    if browser != "auto":
        rows = collect_cookie_rows(browser, target_host, None, profile)
        return browser, build_cookie_header(rows, target_host, request_path, secure=secure)

    errors: list[str] = []
    for candidate in AUTO_BROWSER_ORDER:
        try:
            rows = collect_cookie_rows(candidate, target_host, None, profile)
            header = build_cookie_header(rows, target_host, request_path, secure=secure)
        except Exception as exc:  # noqa: BLE001 - collect the reason and try the next browser
            errors.append(f"{candidate}: {exc}")
            continue
        return candidate, header

    detail = "\n  ".join(errors)
    raise RuntimeError(f"auto-detect found no browser with usable KML cookies:\n  {detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8080)
    parser.add_argument(
        "--browser",
        choices=("auto", "firefox", "chrome", "chromium", "edge", "brave"),
        default="auto",
        help="Which browser's cookies to use (default: auto-detect).",
    )
    parser.add_argument(
        "--profile",
        default="Default",
        help="Chromium profile directory name (default: 'Default'). Ignored for Firefox.",
    )
    parser.add_argument(
        "--cookie-db",
        type=Path,
        help="Explicit cookie DB path, overriding auto-detection for the chosen browser.",
    )
    parser.add_argument("--tunnel-script", type=Path, default=DEFAULT_TUNNEL_SCRIPT)
    parser.add_argument(
        "--tunnel-python",
        default=os.environ.get(
            "LEROBOT_PYTHON",
            str(DEFAULT_LEROBOT_PYTHON) if DEFAULT_LEROBOT_PYTHON.exists() else "python",
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ws_url = urlsplit(args.ws_url)
    if (
        ws_url.scheme != "wss"
        or not ws_url.hostname
        or ws_url.username is not None
        or ws_url.password is not None
        or not ws_url.hostname.endswith(".corp.kuaishou.com")
    ):
        raise ValueError("--ws-url must be a wss://*.corp.kuaishou.com URL without user information")

    target_host = ws_url.hostname
    request_path = ws_url.path or "/"
    secure = ws_url.scheme == "wss"

    if args.cookie_db is not None:
        if args.browser == "auto":
            raise ValueError("--cookie-db requires an explicit --browser (not 'auto')")
        rows = collect_cookie_rows(args.browser, target_host, args.cookie_db, args.profile)
        used_browser = args.browser
        cookie = build_cookie_header(rows, target_host, request_path, secure=secure)
    else:
        used_browser, cookie = resolve_browser(args.browser, target_host, request_path, secure, args.profile)

    print(f"Using cookies from: {used_browser}", file=sys.stderr)

    env = os.environ.copy()
    env["KML_COOKIE"] = cookie

    cmd = [
        args.tunnel_python,
        str(args.tunnel_script),
        "client",
        f"--listen-host={args.listen_host}",
        f"--listen-port={args.listen_port}",
        f"--ws-url={args.ws_url}",
        "--cookie-env=KML_COOKIE",
    ]
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
