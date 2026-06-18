from __future__ import annotations

import logging
import os
import queue
from contextlib import contextmanager
from typing import Any, Generator, List, Optional

import pymysql

from myidea.config import MYSQL_CHARSET, MYSQL_DATABASE, MYSQL_HOST, MYSQL_PASSWORD, MYSQL_PORT, MYSQL_USER

logger = logging.getLogger("myidea.models.database")

# 连接池最大连接数。LIFO 更容易复用“刚用过”的热连接，减少 ping/reconnect 频率。
_POOL_MAX = int(os.environ.get("MYIDEA_DB_POOL_SIZE", "5"))
_POOL: "queue.LifoQueue[pymysql.Connection]" = queue.LifoQueue(maxsize=max(0, _POOL_MAX))


def _get_connection() -> pymysql.Connection:
    """创建一个新的 MySQL 连接（autocommit=True，写日志更方便）。"""
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset=MYSQL_CHARSET,
        autocommit=True,
    )


def _acquire_conn() -> pymysql.Connection:
    """拿一个可用连接：
    - 先从池子里取
    - 取不到就新建
    - 再 ping 一下，断了就自动重连
    """
    if _POOL_MAX <= 0:
        return _get_connection()
    try:
        conn = _POOL.get_nowait()
    except queue.Empty:
        conn = _get_connection()
    try:
        conn.ping(reconnect=True)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        conn = _get_connection()
    return conn


def _release_conn(conn: pymysql.Connection) -> None:
    """把连接放回池子：
    - 如果池子满了，就直接关掉（避免连接越攒越多）
    """
    if _POOL_MAX <= 0:
        conn.close()
        return
    try:
        _POOL.put_nowait(conn)
    except queue.Full:
        conn.close()


@contextmanager
def get_conn() -> Generator[pymysql.Connection, None, None]:
    """获取连接的 with 用法：确保连接最后会被归还（或关闭）。"""
    conn = _acquire_conn()
    try:
        yield conn
    finally:
        _release_conn(conn)


def execute_sql(sql: str, params: tuple = ()) -> int:
    """执行一条写操作 SQL，返回 lastrowid（比如注册用户时插入 users）。"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.lastrowid


def execute_many(sql: str, params_list: list[tuple]) -> int:
    """批量执行 SQL（比如一次性写很多条 impression），返回影响行数。"""
    if not params_list:
        return 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, params_list)
            return cur.rowcount


def fetch_all(sql: str, params: tuple = ()) -> List[tuple]:
    """查询多行数据。"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def fetch_one(sql: str, params: tuple = ()) -> Optional[tuple]:
    """查询单行数据。"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def ensure_database() -> None:
    """确保数据库存在：启动时自动建库，省得你手工点来点去。"""
    conn = pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        charset=MYSQL_CHARSET,
        autocommit=True,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}` "
                f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        logger.info("database '%s' ensured", MYSQL_DATABASE)
    finally:
        conn.close()


def ensure_tables() -> None:
    """确保表结构存在：启动时自动建表，保证一运行就能演示。"""
    from myidea.config import DB_TABLES

    with get_conn() as conn:
        with conn.cursor() as cur:
            for table_name, ddl in DB_TABLES.items():
                cur.execute(ddl)
        logger.info("tables ensured: %s", ", ".join(DB_TABLES.keys()))


def ensure_movies_seed(movies: dict) -> None:
    """把 movies.dat 的电影信息写进 MySQL 的 movies 表（可重复执行，不会重复插入）。

为什么要做这一步：
- 文档里提到了 movies 表，这里把数据补齐，方便验收/讲解
- 后面如果要做“从数据库读电影信息”，就有数据可用
"""
    rows: list[tuple] = []
    for movie_id, movie in movies.items():
        genres = ""
        try:
            genres = "|".join(movie.genres) if getattr(movie, "genres", None) else ""
        except Exception:
            genres = ""
        rows.append((int(movie_id), str(movie.title), genres))

    sql = (
        "INSERT INTO movies(movie_id, title, genres) VALUES (%s, %s, %s) "
        "ON DUPLICATE KEY UPDATE title=VALUES(title), genres=VALUES(genres)"
    )
    affected = execute_many(sql, rows)
    logger.info("movies seeded: %s rows affected", affected)
