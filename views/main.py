from __future__ import annotations

import time

from flask import Blueprint, redirect, render_template, request, session, url_for

from myidea.services.recommendation_service import RecommendationService

main_bp = Blueprint("main", __name__)


def _get_current_user():
    uid = session.get("user_id")
    if uid is None:
        return None
    try:
        return int(uid)
    except (TypeError, ValueError):
        return None


@main_bp.get("/")
def index():
    reco = _get_reco_service()
    user_id = _get_current_user()
    popular = reco.get_popular(top_n=12, exclude_user_id=user_id)
    return render_template(
        "index.html",
        current_user_id=user_id,
        current_username=session.get("username"),
        popular=popular,
    )


@main_bp.get("/me")
def me():
    user_id = _get_current_user()
    if user_id is None:
        return redirect(url_for("auth.login_page"))

    refresh_token = int(time.time() * 1000)
    seed = request.args.get("seed", 0, type=int)

    reco = _get_reco_service()
    history = reco.get_user_history(user_id=user_id, limit=12)
    recs = reco.get_blended_recommendations(user_id=user_id, top_n=10, seed=seed)

    return render_template(
        "me.html",
        current_user_id=user_id,
        current_username=session.get("username"),
        history=history,
        recs=recs,
        refresh_token=refresh_token,
    )


def _get_reco_service() -> RecommendationService:
    from flask import current_app

    return current_app.extensions["reco_service"]
