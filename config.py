from __future__ import annotations

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(BASE_DIR / ".env")

DATASET_DIR = BASE_DIR / "数据集" / "MovieLens" / "ml-1m" / "ml-1m"
ITEMCF_SIM_CACHE_PATH = BASE_DIR / "models" / "itemcf_sim.pkl"

# 未配置时使用进程级随机密钥，避免把可预测的密钥提交到公开仓库。
# 正式部署必须通过 .env 或环境变量提供固定值，否则重启后现有会话会失效。
SECRET_KEY = os.environ.get("MYIDEA_SECRET_KEY") or secrets.token_hex(32)

MYSQL_HOST = os.environ.get("MYIDEA_MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.environ.get("MYIDEA_MYSQL_PORT", "3306"))
MYSQL_USER = os.environ.get("MYIDEA_MYSQL_USER", "root")
MYSQL_PASSWORD = os.environ.get("MYIDEA_MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.environ.get("MYIDEA_MYSQL_DATABASE", "myidea_recommend")
MYSQL_CHARSET = "utf8mb4"

TMDB_API_KEY = os.environ.get("MYIDEA_TMDB_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w185"

GENRE_COLORS = {
    "Action": ("#e53935", "#b71c1c"),
    "Adventure": ("#43a047", "#1b5e20"),
    "Animation": ("#fb8c00", "#e65100"),
    "Children": ("#ffb300", "#ff6f00"),
    "Comedy": ("#fdd835", "#f9a825"),
    "Crime": ("#546e7a", "#263238"),
    "Documentary": ("#78909c", "#37474f"),
    "Drama": ("#5c6bc0", "#283593"),
    "Fantasy": ("#8e24aa", "#4a148c"),
    "Film-Noir": ("#424242", "#212121"),
    "Horror": ("#1e1e1e", "#000000"),
    "Musical": ("#ec407a", "#880e4f"),
    "Mystery": ("#26a69a", "#004d40"),
    "Romance": ("#ef5350", "#c62828"),
    "Sci-Fi": ("#29b6f6", "#01579b"),
    "Thriller": ("#7e57c2", "#311b92"),
    "War": ("#8d6e63", "#3e2723"),
    "Western": ("#a1887f", "#4e342e"),
    "default": ("#607d8b", "#263238"),
}

DB_TABLES = {
    "users": (
        "CREATE TABLE IF NOT EXISTS users ("
        " id INT AUTO_INCREMENT PRIMARY KEY,"
        " username VARCHAR(64) NOT NULL UNIQUE,"
        " password_hash VARCHAR(256) NOT NULL,"
        " created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    ),
    "user_events": (
        "CREATE TABLE IF NOT EXISTS user_events ("
        " id INT AUTO_INCREMENT PRIMARY KEY,"
        " user_id INT NOT NULL,"
        " item_id INT NOT NULL,"
        " event_type VARCHAR(32) NOT NULL,"
        " event_value DOUBLE,"
        " ts BIGINT NOT NULL,"
        " INDEX idx_ue_user_ts (user_id, ts),"
        " INDEX idx_ue_item_ts (item_id, ts)"
        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    ),
    "movies": (
        "CREATE TABLE IF NOT EXISTS movies ("
        " movie_id INT PRIMARY KEY,"
        " title VARCHAR(512) NOT NULL,"
        " genres VARCHAR(256) NOT NULL DEFAULT ''"
        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    ),
    "poster_cache": (
        "CREATE TABLE IF NOT EXISTS poster_cache ("
        " movie_id INT NOT NULL,"
        " poster_url VARCHAR(512) NOT NULL DEFAULT '',"
        " poster_type VARCHAR(32) NOT NULL DEFAULT 'gradient',"
        " PRIMARY KEY (movie_id)"
        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
    ),
}
