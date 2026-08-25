from __future__ import annotations

import logging
import os
import sys
import threading
from datetime import timedelta
from pathlib import Path

from flask import Flask
from flask_wtf.csrf import CSRFProtect

from myidea.config import DATASET_DIR, SECRET_KEY

MYIDEA_DIR = Path(__file__).resolve().parents[1]
if str(MYIDEA_DIR.parent) not in sys.path:
    sys.path.insert(0, str(MYIDEA_DIR.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("myidea.web.app")
csrf = CSRFProtect()


def create_app() -> Flask:
    """创建 Flask app，并完成 MySQL / 数据集 / 路由蓝图的初始化。"""
    app = Flask(
        __name__,
        template_folder=str(MYIDEA_DIR / "templates"),
        static_folder=str(MYIDEA_DIR / "static"),
        static_url_path="/static",
    )
    app.config.update(
        SECRET_KEY=SECRET_KEY,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    )
    csrf.init_app(app)
    # 开发/演示时经常改页面：关掉缓存，避免“我改了但网页没变”的情况
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    app.jinja_env.auto_reload = True

    _init_mysql(app)
    _init_dataset(app)
    _register_blueprints(app)

    logger.info("app created successfully")
    return app


def _init_mysql(app: Flask) -> None:
    """初始化 MySQL：确保数据库与表结构存在（启动时自动建库建表）。"""
    from myidea.models.database import ensure_database, ensure_tables

    try:
        ensure_database()
        ensure_tables()
        logger.info("mysql initialized: database and tables ready")
    except Exception as exc:
        logger.error("mysql init failed: %s", exc)
        raise RuntimeError(
            "MySQL 连接失败，请确认：\n"
            "  1) MySQL 服务已启动\n"
            "  2) 已复制 .env.example 为 .env 并填写数据库密码\n"
            "  3) 也可通过 MYIDEA_MYSQL_* 环境变量覆盖连接配置\n"
            f"  原始错误: {exc}"
        ) from exc


def _init_dataset(app: Flask) -> None:
    """加载 MovieLens 数据集并初始化推荐服务。

补充说明：
- 数据集来自 movies.dat/ratings.dat
- 启动时会把 movies.dat 写入 MySQL 的 movies 表（可重复执行，不会重复插入）
"""
    from myidea.core.data_loader import load_movielens_dat
    from myidea.services.recommendation_service import RecommendationService
    from myidea.models.database import ensure_movies_seed

    dataset = load_movielens_dat(DATASET_DIR)
    reco_service = RecommendationService(dataset)
    app.extensions["reco_service"] = reco_service
    app.extensions["dataset"] = dataset
    ensure_movies_seed(dataset.movies)

    logger.info(
        "dataset loaded: movies=%s ratings=%s users=%s",
        len(dataset.movies),
        len(dataset.ratings),
        len(dataset.user_ratings),
    )

    if os.environ.get("MYIDEA_FAST_START", "0") != "1":
        if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or os.environ.get("MYIDEA_WARMUP_ON_START", "0") == "1":
            # 预热会构建 ItemCF 相似度矩阵/TwoTower item 向量，属于耗时操作
            # 放到后台线程执行：让服务先起来，用户先能打开网页
            t = threading.Thread(target=reco_service.warmup, daemon=True)
            t.start()


def _register_blueprints(app: Flask) -> None:
    """注册蓝图与模板过滤器。"""
    from myidea.views.auth import EVENT_TYPE_LABELS, auth_bp
    from myidea.views.main import main_bp
    from myidea.views.movie import movie_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(movie_bp)

    @app.after_request
    def _disable_cache(response):
        # 禁用缓存：每次都拿到最新页面（对开发/演示更友好）
        if not (response.direct_passthrough or response.status_code in {301, 302, 304}):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    @app.template_filter("datetime")
    def _datetime_filter(ts) -> str:
        from datetime import datetime as dt
        try:
            if ts is None:
                return "-"
            return dt.fromtimestamp(int(ts)).strftime("%Y-%m-%d %H:%M")
        except Exception:
            return "-"

    @app.context_processor
    def _inject_globals():
        return {"event_type_labels": EVENT_TYPE_LABELS}

    logger.info("blueprints registered: auth, main, movie")


app = create_app()
