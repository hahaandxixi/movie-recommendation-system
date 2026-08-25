from __future__ import annotations

import hashlib
import hmac
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from werkzeug.security import check_password_hash, generate_password_hash

from myidea.models.database import fetch_one, execute_sql

logger = logging.getLogger("myidea.services.user_service")
MIN_PASSWORD_LENGTH = 8
_LEGACY_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class User:
    """用户数据对象：基本身份信息"""
    id: int
    username: str
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class UpdateResult:
    """用户资料更新结果：success 表示成功，error 携带失败原因"""
    success: bool
    error: str = ""


def _hash_password(password: str) -> str:
    """生成带盐的自适应密码哈希。"""
    return generate_password_hash(password)


def _verify_password(stored_hash: str, password: str) -> tuple[bool, bool]:
    """返回 (是否匹配, 是否为需要升级的旧 SHA-256 哈希)。"""
    if _LEGACY_SHA256_RE.fullmatch(stored_hash):
        legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        return hmac.compare_digest(stored_hash, legacy_hash), True
    try:
        return check_password_hash(stored_hash, password), False
    except (TypeError, ValueError):
        return False, False


def register(username: str, password: str) -> Optional[User]:
    if not username or len(username.strip()) < 2:
        return None
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return None

    existing = fetch_one("SELECT id FROM users WHERE username=%s", (username.strip(),))
    if existing:
        return None

    pw_hash = _hash_password(password)
    user_id = execute_sql(
        "INSERT INTO users(username, password_hash) VALUES (%s, %s)",
        (username.strip(), pw_hash),
    )
    logger.info("user registered: id=%s username=%s", user_id, username.strip())
    return User(id=user_id, username=username.strip())


def login(username: str, password: str) -> Optional[User]:
    if not username or not password:
        return None

    row = fetch_one(
        "SELECT id, username, password_hash FROM users WHERE username=%s",
        (username.strip(),),
    )
    if row is None:
        return None

    password_matches, is_legacy = _verify_password(row[2], password)
    if not password_matches:
        return None
    if is_legacy:
        execute_sql(
            "UPDATE users SET password_hash=%s WHERE id=%s",
            (_hash_password(password), row[0]),
        )
        logger.info("user %s password hash upgraded", row[0])

    user = User(id=row[0], username=row[1])
    logger.info("user login: id=%s username=%s", user.id, user.username)
    return user


def get_user_by_id(user_id: int) -> Optional[User]:
    row = fetch_one(
        "SELECT id, username, created_at FROM users WHERE id=%s",
        (user_id,),
    )
    if row is None:
        return None
    created_at = row[2] if row[2] is not None else None
    return User(id=row[0], username=row[1], created_at=created_at)


def update_username(user_id: int, new_username: str) -> UpdateResult:
    """修改用户名：需验证长度和唯一性"""
    new_username = new_username.strip()
    if len(new_username) < 2:
        return UpdateResult(success=False, error="用户名至少2个字符")

    existing = fetch_one(
        "SELECT id FROM users WHERE username=%s AND id!=%s",
        (new_username, user_id),
    )
    if existing:
        return UpdateResult(success=False, error="用户名已被占用")

    execute_sql(
        "UPDATE users SET username=%s WHERE id=%s",
        (new_username, user_id),
    )
    logger.info("user %d updated username to '%s'", user_id, new_username)
    return UpdateResult(success=True)


def update_password(user_id: int, old_password: str, new_password: str) -> UpdateResult:
    """修改密码：需验证旧密码正确，新密码至少8个字符。"""
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return UpdateResult(success=False, error="新密码至少8个字符")

    row = fetch_one(
        "SELECT password_hash FROM users WHERE id=%s",
        (user_id,),
    )
    if row is None or not _verify_password(row[0], old_password)[0]:
        return UpdateResult(success=False, error="旧密码错误")

    new_hash = _hash_password(new_password)
    execute_sql(
        "UPDATE users SET password_hash=%s WHERE id=%s",
        (new_hash, user_id),
    )
    logger.info("user %d password updated", user_id)
    return UpdateResult(success=True)
