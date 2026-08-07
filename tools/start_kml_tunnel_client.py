#!/usr/bin/env python3
"""Start the KML WebSocket tunnel client using cookies from local Chrome.

Chrome (or Chromium) must already have a valid KML SSO session:

    python tools/start_kml_tunnel_client.py

The script does not print the cookie. It reads Chrome's cookie DB, decrypts
only the cookies applicable to the KML gateway URL using the key stored in the
OS keyring (Secret Service / libsecret), sets KML_COOKIE in the child process
environment, and execs tools/ws_tcp_tunnel.py client.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_TARGET_HOST = "kml-dtmachine-27353-prod-0.kmlhb2az1l3-2.corp.kuaishou.com"
DEFAULT_WS_URL = f"wss://{DEFAULT_TARGET_HOST}/ws"
# Chrome/Chromium profile roots to probe, in priority order. Each may contain a
# "Default" profile plus numbered "Profile N" profiles.
DEFAULT_CHROME_ROOTS = (
    Path.home() / ".config/google-chrome",
    Path.home() / ".config/chromium",
    Path.home() / "snap/chromium/common/chromium",
    Path.home() / ".var/app/com.google.Chrome/config/google-chrome",
)
# Keyring secret labels/attributes used by Chrome and Chromium on Linux.
CHROME_KEYRING_APPLICATIONS = ("chrome", "chromium")
# Windows FILETIME epoch (1601-01-01) offset from the Unix epoch, in seconds.
_WINDOWS_EPOCH_OFFSET_S = 11644473600
DEFAULT_TUNNEL_SCRIPT = Path(__file__).with_name("ws_tcp_tunnel.py")
DEFAULT_LEROBOT_PYTHON = Path("/home/pnp/miniconda3/envs/lerobot/bin/python")


def find_default_chrome_cookie_db() -> Path:
    for chrome_root in DEFAULT_CHROME_ROOTS:
        if not chrome_root.exists():
            continue
        # Prefer the "Default" profile, then any numbered profile.
        candidates = [chrome_root / "Default" / "Cookies"]
        candidates.extend(sorted(chrome_root.glob("Profile */Cookies")))
        # Newer Chrome stores cookies under a "Network" subdirectory.
        candidates.extend(sorted(chrome_root.glob("*/Network/Cookies")))
        for cookie_db in candidates:
            if cookie_db.exists():
                return cookie_db

    raise FileNotFoundError(
        "Chrome cookies database was not found. Start Chrome and open the KML page at least once."
    )


def _derive_key(password: bytes) -> bytes:
    # Chrome derives an AES-128 key with PBKDF2-HMAC-SHA1, salt "saltysalt", 1 iteration.
    return hashlib.pbkdf2_hmac("sha1", password, b"saltysalt", 1, 16)


def _get_keyring_password() -> bytes:
    """Return the Chrome "Safe Storage" password from the OS keyring.

    Falls back to the well-known "peanuts" password used by the v10 scheme when
    no keyring secret is available.
    """
    try:
        import secretstorage
    except ImportError:
        return b"peanuts"

    try:
        conn = secretstorage.dbus_init()
        collection = secretstorage.get_default_collection(conn)
        if collection.is_locked():
            collection.unlock()
        for application in CHROME_KEYRING_APPLICATIONS:
            # search_items avoids iterating every keyring entry, which can fail
            # if a stale item is deleted mid-iteration.
            for item in collection.search_items({"application": application}):
                try:
                    return item.get_secret()
                except Exception:
                    continue
    except Exception:
        pass
    return b"peanuts"


class _ChromeCookieDecryptor:
    def __init__(self) -> None:
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        self._Cipher = Cipher
        self._algorithms = algorithms
        self._modes = modes
        self._backend = default_backend()
        self._keys = {
            b"v10": _derive_key(b"peanuts"),
            b"v11": _derive_key(_get_keyring_password()),
        }

    def decrypt(self, encrypted_value: bytes, host_key: str) -> str:
        prefix = encrypted_value[:3]
        key = self._keys.get(prefix)
        if key is None:
            # Not an encrypted value (older/plaintext cookie); decode directly.
            return encrypted_value.decode("utf-8", "replace")

        iv = b" " * 16
        cipher = self._Cipher(
            self._algorithms.AES(key), self._modes.CBC(iv), backend=self._backend
        )
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(encrypted_value[3:]) + decryptor.finalize()

        # Strip PKCS7 padding.
        if plaintext:
            pad = plaintext[-1]
            if 1 <= pad <= 16:
                plaintext = plaintext[:-pad]

        # Newer Chrome (domain-bound cookies) prepends the SHA-256 of the host
        # key to the plaintext. Strip it when present.
        if plaintext[:32] == hashlib.sha256(host_key.encode()).digest():
            plaintext = plaintext[32:]

        return plaintext.decode("utf-8", "replace")


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
    hosts: list[str] = []
    for index in range(len(labels) - 1):
        suffix = ".".join(labels[index:])
        hosts.append(suffix)
        hosts.append(f".{suffix}")
    return hosts


def _query_chrome_cookie_rows(
    conn: sqlite3.Connection,
    target_host: str,
) -> list[tuple[str, str, str, bytes, str, int, int]]:
    candidate_hosts = _candidate_hosts(target_host)
    placeholders = ",".join("?" for _ in candidate_hosts)
    rows = conn.execute(
        f"""
        select host_key, name, value, encrypted_value, path, expires_utc, is_secure
        from cookies
        where host_key in ({placeholders})
        order by length(path) desc, creation_utc asc
        """,
        candidate_hosts,
    ).fetchall()
    return rows


def _file_state(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_ino, stat.st_size, stat.st_mtime_ns


def _read_chrome_cookie_rows(
    cookie_db: Path,
    target_host: str,
) -> list[tuple[str, str, str, bytes, str, int, int]]:
    if not cookie_db.exists():
        raise FileNotFoundError(f"Chrome cookie DB not found: {cookie_db}")

    live_uri = f"{cookie_db.resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(live_uri, uri=True, timeout=2.0)
        try:
            conn.execute("pragma query_only=on")
            conn.execute("begin")
            return _query_chrome_cookie_rows(conn, target_host)
        finally:
            conn.close()
    except sqlite3.OperationalError as exc:
        if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
            raise

    # Chrome holds an exclusive lock while running. Fall back to a stable
    # main+WAL copy; file metadata is checked before and after every copy to
    # reject racing snapshots.
    wal = Path(f"{cookie_db}-wal")
    with tempfile.TemporaryDirectory(prefix="kml_chrome_cookies_") as tmp_dir:
        for attempt in range(5):
            before = (_file_state(cookie_db), _file_state(wal))
            attempt_dir = Path(tmp_dir) / str(attempt)
            attempt_dir.mkdir()
            snapshot = attempt_dir / "Cookies"
            try:
                shutil.copy2(cookie_db, snapshot)
                if wal.exists():
                    shutil.copy2(wal, Path(f"{snapshot}-wal"))
            except FileNotFoundError:
                continue
            if before != (_file_state(cookie_db), _file_state(wal)):
                continue

            conn = sqlite3.connect(snapshot)
            try:
                conn.execute("pragma query_only=on")
                return _query_chrome_cookie_rows(conn, target_host)
            except sqlite3.DatabaseError:
                continue
            finally:
                conn.close()

    raise RuntimeError("Could not obtain a consistent Chrome cookie database snapshot")


def build_cookie_header(cookie_db: Path, target_host: str, request_path: str, *, secure: bool) -> str:
    now = int(time.time())
    cookie_parts: list[str] = []
    rows = _read_chrome_cookie_rows(cookie_db, target_host)
    decryptor = _ChromeCookieDecryptor()
    for host, name, value, encrypted_value, path, expires_utc, is_secure in rows:
        if not _domain_matches(host, target_host) or not _path_matches(path, request_path):
            continue
        # expires_utc is microseconds since 1601-01-01; 0 means a session cookie.
        if expires_utc:
            expiry_seconds = expires_utc / 1_000_000 - _WINDOWS_EPOCH_OFFSET_S
            if expiry_seconds <= now:
                continue
        if is_secure and not secure:
            continue

        if value:
            cookie_value = value
        elif encrypted_value:
            cookie_value = decryptor.decrypt(encrypted_value, host)
        else:
            continue
        if cookie_value is None:
            continue

        if not name or any(
            character in ";=" or ord(character) < 0x20 or ord(character) == 0x7F
            for character in name
        ):
            continue
        if any(character == ";" or ord(character) < 0x20 or ord(character) == 0x7F for character in cookie_value):
            continue

        # The query is already in RFC cookie-header order: longer paths first,
        # then earlier creation time. Preserve duplicate names and this exact order.
        cookie_parts.append(f"{name}={cookie_value}")

    if not cookie_parts:
        raise RuntimeError(
            "No usable KML cookies were found. Open the KML page in Chrome, complete SSO login, "
            "then run this script again."
        )

    return "; ".join(cookie_parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws-url", default=DEFAULT_WS_URL)
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8080)
    parser.add_argument(
        "--cookie-db",
        type=Path,
        help="Chrome Cookies database (default: auto-detect the default Chrome or Chromium profile)",
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
    cookie_db = args.cookie_db or find_default_chrome_cookie_db()
    cookie = build_cookie_header(
        cookie_db,
        ws_url.hostname,
        ws_url.path or "/",
        secure=ws_url.scheme == "wss",
    )

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
