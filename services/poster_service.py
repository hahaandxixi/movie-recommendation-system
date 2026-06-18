from __future__ import annotations

import logging
import re
import time
import urllib.request
import urllib.parse
import urllib.error
import json
import gzip
from pathlib import Path
from typing import Dict, Optional

from myidea.config import GENRE_COLORS, TMDB_API_KEY, TMDB_BASE_URL, TMDB_IMAGE_BASE
from myidea.models.database import execute_sql, fetch_one

logger = logging.getLogger("myidea.services.poster_service")

_STATIC_DIR = Path(__file__).resolve().parents[1] / "static" / "posters"
_MEM_CACHE: Dict[int, str] = {}


def get_poster_url(movie_id: int, title: str = "", genres: str = "") -> str:
    """拿电影海报 URL：有就返回，没有就返回空字符串（页面会用占位图兜底）。"""
    cached = _MEM_CACHE.get(movie_id)
    if cached:
        return cached

    # MySQL 缓存：重启后也能复用
    db_cached = _check_db_cache(movie_id)
    if db_cached:
        _MEM_CACHE[movie_id] = db_cached
        return db_cached

    # 本地海报：无需网络（最快）
    local = _check_local(movie_id)
    if local:
        _upsert_cache(movie_id, local, "local")
        _MEM_CACHE[movie_id] = local
        return local

    # TMDB（可选）：需要 key；网络失败就当作没海报，继续用占位图
    tmdb = _check_tmdb(title=title)
    if tmdb:
        _upsert_cache(movie_id, tmdb, "tmdb")
        _MEM_CACHE[movie_id] = tmdb
        return tmdb

    return ""


def _check_local(movie_id: int) -> Optional[str]:
    f = _STATIC_DIR / f"{movie_id}.jpg"
    if f.exists() and f.stat().st_size > 500:
        return f"/static/posters/{movie_id}.jpg"
    return None


def _check_db_cache(movie_id: int) -> Optional[str]:
    try:
        row = fetch_one(
            "SELECT poster_url FROM poster_cache WHERE movie_id=%s",
            (movie_id,),
        )
        if row and row[0]:
            return str(row[0])
    except Exception as exc:
        logger.debug("poster_cache lookup failed: %s", exc)
    return None


def _upsert_cache(movie_id: int, poster_url: str, poster_type: str) -> None:
    try:
        execute_sql(
            "INSERT INTO poster_cache(movie_id, poster_url, poster_type) VALUES (%s, %s, %s) "
            "ON DUPLICATE KEY UPDATE poster_url=VALUES(poster_url), poster_type=VALUES(poster_type)",
            (movie_id, poster_url, poster_type),
        )
    except Exception as exc:
        logger.debug("poster_cache upsert failed: %s", exc)


def _check_tmdb(title: str) -> Optional[str]:
    if not TMDB_API_KEY or not title:
        return None

    try:
        clean_title, year = _split_title_year(title)
        poster_path = _tmdb_search_poster(clean_title, year=year)
        if not poster_path:
            return None
        return f"{TMDB_IMAGE_BASE}{poster_path}"
    except Exception as exc:
        logger.debug("tmdb search failed: %s", exc)
        return None


def _split_title_year(title: str) -> tuple[str, str]:
    m = re.search(r"\((\d{4})\)\s*$", title)
    if not m:
        return title.strip(), ""
    year = m.group(1)
    clean = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    return clean, year


def _tmdb_search_poster(title: str, year: str = "") -> str:
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "include_adult": "false",
    }
    if year:
        params["year"] = year

    url = f"{TMDB_BASE_URL}/search/movie?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=4) as resp:
        raw = resp.read()

    data = json.loads(raw.decode("utf-8", errors="replace"))
    results = data.get("results") or []
    for r in results:
        poster_path = r.get("poster_path") or ""
        if poster_path:
            return poster_path
    return ""


def get_genre_colors(genres: str) -> tuple:
    genre_list = [g.strip() for g in genres.split("|") if g.strip()]
    primary_genre = genre_list[0] if genre_list else "default"
    return GENRE_COLORS.get(primary_genre, GENRE_COLORS["default"])
