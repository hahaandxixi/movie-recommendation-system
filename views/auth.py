from __future__ import annotations

from flask import Blueprint, redirect, render_template, request, session, url_for

from myidea.services.interaction_service import get_user_history_paginated, get_user_event_stats
from myidea.services.user_service import (
    login as user_login,
    register as user_register,
    get_user_by_id,
    update_username,
    update_password,
)

auth_bp = Blueprint("auth", __name__)

EVENT_TYPE_LABELS = {
    "click": "点击",
    "rate": "评分",
    "like": "收藏",
    "impression": "曝光",
}


def _get_current_user_id():
    """从 session 获取当前登录用户 ID，未登录返回 None"""
    uid = session.get("user_id")
    if uid is None:
        return None
    try:
        return int(uid)
    except (TypeError, ValueError):
        return None


@auth_bp.get("/login")
def login_page():
    if session.get("user_id"):
        return redirect(url_for("main.me"))
    return render_template("login.html")


@auth_bp.post("/login")
def login_submit():
    username = (request.form.get("username") or "").strip()
    password = (request.form.get("password") or "").strip()

    if not username:
        return render_template("login.html", error="请输入用户名")
    if not password:
        return render_template("login.html", error="请输入密码")

    user = user_login(username, password)
    if user is None:
        return render_template("login.html", error="用户名或密码错误")

    session.clear()
    session["user_id"] = user.id
    session["username"] = user.username
    return redirect(url_for("main.me"))


@auth_bp.get("/register")
def register_page():
    if session.get("user_id"):
        return redirect(url_for("main.me"))
    return render_template("register.html")


@auth_bp.post("/register")
def register_submit():
    username = (request.form.get("username") or "").strip()
    password = (request.form.get("password") or "").strip()
    password2 = (request.form.get("password2") or "").strip()

    error = None
    if len(username) < 2:
        error = "用户名至少2个字符"
    elif not username.isalnum() and not all(c.isalnum() or c in "_-" for c in username):
        error = "用户名只能包含字母、数字、下划线和连字符"
    elif len(password) < 8:
        error = "密码至少8个字符"
    elif password != password2:
        error = "两次密码不一致"

    if error:
        return render_template("register.html", error=error)

    user = user_register(username, password)
    if user is None:
        return render_template("register.html", error="用户名已存在或注册失败")

    session.clear()
    session["user_id"] = user.id
    session["username"] = user.username
    return redirect(url_for("main.me"))


@auth_bp.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("main.index"))


@auth_bp.get("/profile")
def profile_page():
    user_id = _get_current_user_id()
    if user_id is None:
        return redirect(url_for("auth.login_page"))

    user = get_user_by_id(user_id)
    stats = get_user_event_stats(user_id)
    return render_template(
        "profile.html",
        current_user_id=user_id,
        current_username=session.get("username"),
        user=user,
        stats=stats,
    )


@auth_bp.post("/profile/username")
def profile_update_username():
    user_id = _get_current_user_id()
    if user_id is None:
        return redirect(url_for("auth.login_page"))

    new_username = (request.form.get("username") or "").strip()
    result = update_username(user_id, new_username)

    if result.success:
        session["username"] = new_username
        return redirect(url_for("auth.profile_page"))

    user = get_user_by_id(user_id)
    stats = get_user_event_stats(user_id)
    return render_template(
        "profile.html",
        current_user_id=user_id,
        current_username=session.get("username"),
        user=user,
        stats=stats,
        error=result.error,
    )


@auth_bp.post("/profile/password")
def profile_update_password():
    user_id = _get_current_user_id()
    if user_id is None:
        return redirect(url_for("auth.login_page"))

    old_password = (request.form.get("old_password") or "").strip()
    new_password = (request.form.get("new_password") or "").strip()
    new_password2 = (request.form.get("new_password2") or "").strip()

    error = None
    if not old_password:
        error = "请输入旧密码"
    elif len(new_password) < 8:
        error = "新密码至少8个字符"
    elif new_password != new_password2:
        error = "两次新密码不一致"

    if error:
        user = get_user_by_id(user_id)
        stats = get_user_event_stats(user_id)
        return render_template(
            "profile.html",
            current_user_id=user_id,
            current_username=session.get("username"),
            user=user,
            stats=stats,
            error=error,
        )

    result = update_password(user_id, old_password, new_password)
    if result.success:
        return redirect(url_for("auth.profile_page"))

    user = get_user_by_id(user_id)
    stats = get_user_event_stats(user_id)
    return render_template(
        "profile.html",
        current_user_id=user_id,
        current_username=session.get("username"),
        user=user,
        stats=stats,
        error=result.error,
    )


@auth_bp.get("/history")
def history_page():
    user_id = _get_current_user_id()
    if user_id is None:
        return redirect(url_for("auth.login_page"))

    page = request.args.get("page", 1, type=int)
    event_type = request.args.get("type", None, type=str)
    if event_type and event_type not in EVENT_TYPE_LABELS:
        event_type = None

    result = get_user_history_paginated(
        user_id=user_id, page=page, per_page=20, event_type=event_type
    )

    from myidea.services.recommendation_service import RecommendationService

    reco = _get_reco_service()
    enriched_events = []
    for e in result["events"]:
        info = reco.get_movie_info(e["item_id"])
        title = info["title"] if info else f"movie_id={e['item_id']}"
        enriched_events.append({**e, "title": title})

    return render_template(
        "history.html",
        current_user_id=user_id,
        current_username=session.get("username"),
        events=enriched_events,
        event_type_labels=EVENT_TYPE_LABELS,
        pagination={
            "page": result["page"],
            "total_pages": result["total_pages"],
            "total": result["total"],
            "per_page": result["per_page"],
        },
        filter_type=event_type or "",
    )


def _get_reco_service():
    from flask import current_app
    return current_app.extensions["reco_service"]
