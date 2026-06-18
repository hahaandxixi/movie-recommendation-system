"""从 TMDB 网站爬取电影海报并缓存到本地 static/posters/"""
import argparse
import urllib.request
import urllib.parse
import urllib.error
import re
import os
import gzip
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("poster_builder")

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static" / "posters"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR = BASE_DIR / "数据集" / "MovieLens" / "ml-1m" / "ml-1m"
TMDB_SEARCH = "https://www.themoviedb.org/search?query="
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}
RATE_LIMIT = 1.0


def _read_lines(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def load_movie_list(limit=None):
    movies_path = DATASET_DIR / "movies.dat"
    movies = []
    for line in _read_lines(movies_path):
        parts = line.split("::")
        if len(parts) >= 3:
            movies.append((int(parts[0]), parts[1].strip()))
    return movies[:limit]


def _load_popular_movie_ids(top_n: int) -> list[int]:
    ratings_path = DATASET_DIR / "ratings.dat"
    counts = {}
    sums = {}
    for line in _read_lines(ratings_path):
        parts = line.split("::")
        if len(parts) < 3:
            continue
        movie_id = int(parts[1])
        rating = float(parts[2])
        counts[movie_id] = counts.get(movie_id, 0) + 1
        sums[movie_id] = sums.get(movie_id, 0.0) + rating

    scored = []
    for mid, cnt in counts.items():
        avg = sums[mid] / cnt
        score = cnt * 0.7 + avg * 0.3
        scored.append((score, mid))
    scored.sort(reverse=True)
    return [mid for _, mid in scored[:top_n]]


def download_missing_for_ids(movie_ids: list[int]) -> tuple[int, int]:
    movies = dict(load_movie_list(limit=None))
    total = len(movie_ids)
    success = 0
    for i, mid in enumerate(movie_ids):
        title = movies.get(mid, "")
        out_path = STATIC_DIR / f"{mid}.jpg"
        if out_path.exists() and out_path.stat().st_size > 500:
            success += 1
            continue
        url = search_tmdb_poster(title) if title else None
        if url:
            if download_poster(mid, url):
                success += 1
        if i < total - 1:
            time.sleep(RATE_LIMIT)
        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d, downloaded: %d", i + 1, total, success)
    logger.info("Done: %d/%d posters downloaded", success, total)
    return success, total



def _urlopen(url, timeout=20):
    req = urllib.request.Request(url, headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=timeout)
    if resp.headers.get("Content-Encoding") == "gzip":
        content = gzip.decompress(content)
    return content


def search_tmdb_poster(title):
    clean = re.sub(r"\(\d{4}\)", "", title).strip()
    year_match = re.search(r"\((\d{4})\)", title)
    year = year_match.group(1) if year_match else ""

    query = urllib.parse.quote(clean)
    url = f"{TMDB_SEARCH}{query}"
    try:
        html = _urlopen(url, timeout=20).decode("utf-8", errors="replace")
        posters = re.findall(r"/t/p/w\d+[a-zA-Z0-9/_.-]+\.jpg", html)
        if posters:
            return "https://www.themoviedb.org" + posters[0].replace("/w92", "/w342")
    except Exception as e:
        logger.debug("Search failed for %s: %s", title, e)
    return None


def download_poster(movie_id, poster_url):
    out_path = STATIC_DIR / f"{movie_id}.jpg"
    if out_path.exists() and out_path.stat().st_size > 500:
        return True
    try:
        img_data = _urlopen(poster_url, timeout=30)
        with open(out_path, "wb") as f:
            f.write(img_data)
        size = len(img_data)
        if size < 500:
            out_path.unlink(missing_ok=True)
            return False
        logger.info("Downloaded: %s -> %s (%d bytes)", movie_id, poster_url[:80], size)
        return True
    except Exception as e:
        logger.debug("Download failed for %s: %s", movie_id, e)
        return False


def build_posters(movie_limit=50, start_id=None, end_id=None):
    movies = load_movie_list(movie_limit)
    if start_id is not None or end_id is not None:
        movies = [(mid, title) for mid, title in movies if (start_id is None or mid >= start_id) and (end_id is None or mid <= end_id)]
    total = len(movies)
    success = 0
    for i, (mid, title) in enumerate(movies):
        out_path = STATIC_DIR / f"{mid}.jpg"
        if out_path.exists() and out_path.stat().st_size > 500:
            success += 1
            continue
        url = search_tmdb_poster(title)
        if url:
            if download_poster(mid, url):
                success += 1
        if i < total - 1:
            time.sleep(RATE_LIMIT)
        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d, downloaded: %d", i + 1, total, success)
    logger.info("Done: %d/%d posters downloaded", success, total)
    return success, total


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--popular", type=int, default=0)
    p.add_argument("--movie-limit", type=int, default=60)
    p.add_argument("--start-id", type=int, default=None)
    p.add_argument("--end-id", type=int, default=None)
    p.add_argument("--rate-limit", type=float, default=RATE_LIMIT)
    args = p.parse_args()

    RATE_LIMIT = float(args.rate_limit)
    if args.popular and args.popular > 0:
        ids = _load_popular_movie_ids(args.popular)
        download_missing_for_ids(ids)
    else:
        build_posters(movie_limit=args.movie_limit, start_id=args.start_id, end_id=args.end_id)
