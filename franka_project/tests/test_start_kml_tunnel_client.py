from __future__ import annotations

import importlib.util
import sqlite3
import sys
import time
from http.cookies import SimpleCookie
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools/start_kml_tunnel_client.py"
SPEC = importlib.util.spec_from_file_location("start_kml_tunnel_client_under_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


def _create_cookie_db(
    path: Path,
    rows: list[tuple[str, str, str, str, int, int, int]],
    *,
    user_version: int = 15,
) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(f"pragma user_version = {user_version}")
        connection.execute(
            """
            create table moz_cookies (
                host text not null,
                name text not null,
                value text not null,
                path text not null,
                expiry integer not null,
                isSecure integer not null,
                creationTime integer not null,
                originAttributes text not null,
                isPartitionedAttributeSet integer not null
            )
            """
        )
        connection.executemany(
            """
            insert into moz_cookies (
                host, name, value, path, expiry, isSecure, creationTime,
                originAttributes, isPartitionedAttributeSet
            ) values (?, ?, ?, ?, ?, ?, ?, '', 0)
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def _parse_cookie_header(header: str) -> dict[str, str]:
    parsed = SimpleCookie()
    parsed.load(header)
    return {name: morsel.value for name, morsel in parsed.items()}


def test_find_default_firefox_cookie_db_uses_default_profile_under_configured_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firefox_root = tmp_path / "isolated-firefox-root"
    non_default_profile = firefox_root / "Profiles/non-default"
    default_profile = firefox_root / "Profiles/default-release"
    non_default_profile.mkdir(parents=True)
    default_profile.mkdir(parents=True)
    (non_default_profile / "cookies.sqlite").touch()
    (default_profile / "cookies.sqlite").touch()
    (firefox_root / "profiles.ini").write_text(
        """
[Profile0]
Name=non-default
IsRelative=1
Path=Profiles/non-default
Default=0

[Profile1]
Name=default-release
IsRelative=1
Path=Profiles/default-release
Default=1
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "DEFAULT_FIREFOX_ROOTS", (firefox_root,))

    assert launcher.find_default_firefox_cookie_db() == default_profile / "cookies.sqlite"


def test_find_default_firefox_cookie_db_prefers_install_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    firefox_root = tmp_path / "isolated-firefox-root"
    profile_default = firefox_root / "Profiles/profile-default"
    install_default = firefox_root / "Profiles/install-default"
    profile_default.mkdir(parents=True)
    install_default.mkdir(parents=True)
    (profile_default / "cookies.sqlite").touch()
    (install_default / "cookies.sqlite").touch()
    (firefox_root / "profiles.ini").write_text(
        """
[Install0123456789ABCDEF]
Default=Profiles/install-default
Locked=1

[Profile0]
Name=default
IsRelative=1
Path=Profiles/profile-default
Default=1
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "DEFAULT_FIREFOX_ROOTS", (firefox_root,))

    assert launcher.find_default_firefox_cookie_db() == install_default / "cookies.sqlite"


def test_build_cookie_header_filters_rows_and_preserves_rfc_order_and_duplicate_names(
    tmp_path: Path,
) -> None:
    target_host = "kml-machine.corp.kuaishou.com"
    request_path = "/ws/control"
    now = int(time.time())
    cookie_db = tmp_path / "cookies.sqlite"
    _create_cookie_db(
        cookie_db,
        [
            (target_host, "exact", "exact-value", "/ws", now + 3600, 0, 100),
            (".corp.kuaishou.com", "parent", "parent-value", "/", 0, 0, 101),
            (".attacker.example", "wrong_domain", "reject-domain", "/ws", now + 3600, 0, 102),
            (target_host, "wrong_path", "reject-path", "/admin", now + 3600, 0, 103),
            (target_host, "path_boundary", "reject-boundary", "/w", now + 3600, 0, 104),
            (target_host, "expired", "reject-expired", "/ws", now - 1, 0, 105),
            (target_host, "secure_only", "secure-value", "/ws", now + 3600, 1, 106),
            (".corp.kuaishou.com", "shared", "broad-value", "/", now + 3600, 0, 107),
            (target_host, "shared", "specific-value", "/ws", now + 3600, 0, 108),
            (target_host, "shared", "later-same-path-value", "/ws", now + 3600, 0, 109),
        ],
    )

    secure_header = launcher.build_cookie_header(
        cookie_db,
        target_host,
        request_path,
        secure=True,
    )
    assert secure_header.split("; ") == [
        "exact=exact-value",
        "secure_only=secure-value",
        "shared=specific-value",
        "shared=later-same-path-value",
        "parent=parent-value",
        "shared=broad-value",
    ]

    insecure_header = launcher.build_cookie_header(
        cookie_db,
        target_host,
        request_path,
        secure=False,
    )
    assert insecure_header.split("; ") == [
        "exact=exact-value",
        "shared=specific-value",
        "shared=later-same-path-value",
        "parent=parent-value",
        "shared=broad-value",
    ]


@pytest.mark.parametrize("user_version", [15, 16])
def test_build_cookie_header_interprets_expiry_units_by_firefox_schema(
    tmp_path: Path,
    user_version: int,
) -> None:
    target_host = "kml-machine.corp.kuaishou.com"
    now = int(time.time())
    multiplier = 1000 if user_version >= 16 else 1
    cookie_db = tmp_path / f"cookies-v{user_version}.sqlite"
    _create_cookie_db(
        cookie_db,
        [
            (
                target_host,
                "future",
                f"future-v{user_version}",
                "/ws",
                (now + 3600) * multiplier,
                0,
                100,
            ),
            (
                target_host,
                "expired",
                f"expired-v{user_version}",
                "/ws",
                (now - 1) * multiplier,
                0,
                101,
            ),
        ],
        user_version=user_version,
    )

    header = launcher.build_cookie_header(cookie_db, target_host, "/ws", secure=True)

    assert _parse_cookie_header(header) == {"future": f"future-v{user_version}"}


def test_build_cookie_header_preserves_empty_cookie_value(tmp_path: Path) -> None:
    target_host = "kml-machine.corp.kuaishou.com"
    cookie_db = tmp_path / "cookies-empty-value.sqlite"
    _create_cookie_db(
        cookie_db,
        [(target_host, "empty_session", "", "/ws", 0, 1, 100)],
        user_version=16,
    )

    header = launcher.build_cookie_header(cookie_db, target_host, "/ws", secure=True)

    assert header == "empty_session="
    assert _parse_cookie_header(header) == {"empty_session": ""}
