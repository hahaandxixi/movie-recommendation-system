import hashlib

from werkzeug.security import check_password_hash, generate_password_hash

from myidea.services import user_service


def test_register_stores_adaptive_password_hash(monkeypatch):
    captured = {}
    monkeypatch.setattr(user_service, "fetch_one", lambda *_: None)

    def fake_execute(sql, params):
        captured["sql"] = sql
        captured["params"] = params
        return 7

    monkeypatch.setattr(user_service, "execute_sql", fake_execute)

    user = user_service.register("alice", "strong-pass")

    assert user is not None and user.id == 7
    assert captured["params"][0] == "alice"
    assert check_password_hash(captured["params"][1], "strong-pass")


def test_login_upgrades_legacy_sha256_hash(monkeypatch):
    legacy = hashlib.sha256(b"strong-pass").hexdigest()
    updates = []
    monkeypatch.setattr(
        user_service,
        "fetch_one",
        lambda *_: (3, "alice", legacy),
    )
    monkeypatch.setattr(user_service, "execute_sql", lambda sql, params: updates.append((sql, params)))

    user = user_service.login("alice", "strong-pass")

    assert user is not None and user.id == 3
    assert len(updates) == 1
    assert check_password_hash(updates[0][1][0], "strong-pass")


def test_update_password_verifies_old_hash(monkeypatch):
    old_hash = generate_password_hash("old-password")
    updates = []
    monkeypatch.setattr(user_service, "fetch_one", lambda *_: (old_hash,))
    monkeypatch.setattr(user_service, "execute_sql", lambda sql, params: updates.append((sql, params)))

    result = user_service.update_password(5, "old-password", "new-password")

    assert result.success
    assert check_password_hash(updates[0][1][0], "new-password")


def test_short_password_is_rejected_without_database_access(monkeypatch):
    monkeypatch.setattr(
        user_service,
        "fetch_one",
        lambda *_: (_ for _ in ()).throw(AssertionError("database should not be called")),
    )

    assert user_service.register("alice", "short") is None
