from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from myidea.services.interaction_service import log_event
from myidea.services.recommendation_service import RecommendationService

movie_bp = Blueprint("movie", __name__, url_prefix="/movie")


def _get_current_user() -> int | None:
    """安全地从 session 中获取当前用户 ID"""
    uid = session.get("user_id")
    if uid is None:
        return None
    try:
        return int(uid)
    except (TypeError, ValueError):
        session.clear()
        return None


@movie_bp.get("/<int:item_id>")
def detail(item_id: int):
    user_id = _get_current_user()
    if user_id is not None:
        log_event(user_id=user_id, item_id=item_id, event_type="click")

    reco = _get_reco_service()
    info = reco.get_movie_info(item_id)
    if info is None:
        return render_template(
            "base.html",
            current_user_id=user_id,
            current_username=session.get("username"),
        ), 404

    similar = reco.get_similar_items(item_id=item_id, top_n=12)

    return render_template(
        "movie_detail.html",
        current_user_id=user_id,
        current_username=session.get("username"),
        movie=info,
        similar=similar,
    )


@movie_bp.post("/<int:item_id>/rate")
def rate(item_id: int):
    user_id = _get_current_user()
    if user_id is None:
        flash("请先登录后再评分", "error")
        return redirect(url_for("auth.login_page"))

    reco = _get_reco_service()
    if reco.get_movie_info(item_id) is None:
        flash("电影不存在", "error")
        return redirect(url_for("main.index"))

    raw = (request.form.get("rating") or "").strip()
    try:
        rating = float(raw)
    except ValueError:
        flash("无效的评分值", "error")
        return redirect(url_for("movie.detail", item_id=item_id))

    rating = round(rating)
    if rating < 1 or rating > 5:
        flash("评分范围是 1-5 分", "error")
        return redirect(url_for("movie.detail", item_id=item_id))

    log_event(user_id=user_id, item_id=item_id, event_type="rate", event_value=rating)
    flash(f"已评分 {int(rating)} 分！", "success")
    next_url = (request.form.get("next") or "").strip()
    if next_url.startswith("/"):
        return redirect(next_url)
    return redirect(url_for("movie.detail", item_id=item_id))


@movie_bp.post("/<int:item_id>/like")
def like(item_id: int):
    user_id = _get_current_user()
    if user_id is None:
        flash("请先登录后再收藏", "error")
        return redirect(url_for("auth.login_page"))

    reco = _get_reco_service()
    if reco.get_movie_info(item_id) is None:
        flash("电影不存在", "error")
        return redirect(url_for("main.index"))

    log_event(user_id=user_id, item_id=item_id, event_type="like", event_value=5.0)
    flash("已收藏！", "success")
    next_url = (request.form.get("next") or "").strip()
    if next_url.startswith("/"):
        return redirect(next_url)
    return redirect(url_for("movie.detail", item_id=item_id))


def _get_reco_service() -> RecommendationService:
    from flask import current_app

    return current_app.extensions["reco_service"]
