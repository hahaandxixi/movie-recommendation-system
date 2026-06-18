from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from myidea.models.database import fetch_one, execute_sql

logger = logging.getLogger("myidea.services.user_service")


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
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def register(username: str, password: str) -> Optional[User]:
    if not username or len(username.strip()) < 2:
        return None
    if not password or len(password) < 3:
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

    pw_hash = _hash_password(password)
    row = fetch_one(
        "SELECT id, username FROM users WHERE username=%s AND password_hash=%s",
        (username.strip(), pw_hash),
    )
    if row is None:
        return None

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
    """修改密码：需验证旧密码正确，新密码至少3个字符"""
    if len(new_password) < 3:
        return UpdateResult(success=False, error="新密码至少3个字符")

    old_hash = _hash_password(old_password)
    row = fetch_one(
        "SELECT id FROM users WHERE id=%s AND password_hash=%s",
        (user_id, old_hash),
    )
    if row is None:
        return UpdateResult(success=False, error="旧密码错误")

    new_hash = _hash_password(new_password)
    execute_sql(
        "UPDATE users SET password_hash=%s WHERE id=%s",
        (new_hash, user_id),
    )
    logger.info("user %d password updated", user_id)
    return UpdateResult(success=True)
