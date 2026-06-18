from __future__ import annotations

import logging
import time
from typing import Dict, List

from myidea.models.database import execute_many, execute_sql, fetch_all, fetch_one

logger = logging.getLogger("myidea.services.interaction_service")


def log_event(user_id: int, item_id: int, event_type: str, event_value: float | None = None) -> None:
    """写入一条交互事件（click/rate/like/impression）。"""
    ts = int(time.time())
    execute_sql(
        "INSERT INTO user_events(user_id, item_id, event_type, event_value, ts) VALUES (%s, %s, %s, %s, %s)",
        (user_id, item_id, event_type, event_value, ts),
    )
    logger.debug("event %s user=%s item=%s value=%s", event_type, user_id, item_id, event_value)


def log_impressions(user_id: int, item_ids: List[int]) -> None:
    """批量写入曝光事件，避免一页推荐写很多次 INSERT 导致变慢。"""
    if not item_ids:
        return
    ts = int(time.time())
    rows = [(user_id, int(item_id), "impression", None, ts) for item_id in item_ids]
    execute_many(
        "INSERT INTO user_events(user_id, item_id, event_type, event_value, ts) VALUES (%s, %s, %s, %s, %s)",
        rows,
    )


def get_recent_events(user_id: int, limit: int = 30) -> List[dict]:
    """读取最近的交互事件（用于 /me 页面展示）。"""
    rows = fetch_all(
        "SELECT item_id, event_type, event_value, ts FROM user_events WHERE user_id=%s ORDER BY ts DESC LIMIT %s",
        (user_id, limit),
    )
    out: List[dict] = []
    for item_id, event_type, event_value, ts in rows:
        out.append(
            {
                "item_id": int(item_id),
                "event_type": str(event_type),
                "event_value": float(event_value) if event_value is not None else None,
                "ts": int(ts),
            }
        )
    return out


def get_user_profile_from_events(user_id: int, max_events: int = 50) -> Dict[int, float]:
    """根据交互日志构建用户画像（越新的行为权重越高）。"""
    rows = fetch_all(
        "SELECT item_id, event_type, event_value, ts FROM user_events WHERE user_id=%s ORDER BY ts DESC LIMIT %s",
        (user_id, max_events),
    )

    now = int(time.time())
    profile: Dict[int, float] = {}

    for item_id, event_type, event_value, ts in rows:
        # 时间衰减：越新的行为越重要（满足“推荐随时间变化”）
        age_days = max(0.0, (now - int(ts)) / 86400.0)
        recency = 1.0 / (1.0 + age_days)
        it = int(item_id)
        et = str(event_type)

        if et == "rate" and event_value is not None:
            # 显式反馈：评分越高代表越喜欢（1~5）
            profile[it] = max(profile.get(it, 0.0), float(event_value)) * recency
        elif et == "click":
            # 隐式反馈：点击算是“可能喜欢”，给一个中等权重
            profile[it] = max(profile.get(it, 0.0), 3.0) * recency
        elif et in {"like", "collect", "favorite"}:
            # 强隐式反馈：收藏基本就是“很喜欢”
            profile[it] = max(profile.get(it, 0.0), 5.0) * recency

    return profile


def get_user_interacted_items(user_id: int) -> set:
    """获取用户已交互过的 item_id 集合，用于推荐去重与过滤。"""
    rows = fetch_all(
        "SELECT DISTINCT item_id FROM user_events WHERE user_id=%s AND event_type IN ('click','rate','like','collect','favorite')",
        (user_id,),
    )
    return {int(r[0]) for r in rows}


def get_user_history_paginated(
    user_id: int, page: int = 1, per_page: int = 20, event_type: str | None = None
) -> dict:
    """分页查询用户交互历史（"个人历史订单信息查询"）"""
    # 老师文档提到“历史订单查询”，这里按课设语境映射为“用户交互历史记录”
    offset = max(0, (page - 1)) * per_page

    base_where = "user_id=%s"
    base_params: list = [user_id]

    if event_type:
        base_where += " AND event_type=%s"
        base_params.append(event_type)

    count_row = fetch_one(
        f"SELECT COUNT(*) FROM user_events WHERE {base_where}", tuple(base_params)
    )
    total = int(count_row[0]) if count_row else 0

    rows = fetch_all(
        f"SELECT item_id, event_type, event_value, ts FROM user_events WHERE {base_where} ORDER BY ts DESC LIMIT %s OFFSET %s",
        tuple(base_params + [per_page, offset]),
    )

    events: list = []
    for item_id, etype, evalue, ts in rows:
        events.append(
            {
                "item_id": int(item_id),
                "event_type": str(etype),
                "event_value": float(evalue) if evalue is not None else None,
                "ts": int(ts),
            }
        )

    total_pages = max(1, (total + per_page - 1) // per_page)
    return {
        "events": events,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
    }


def get_user_event_stats(user_id: int) -> dict:
    """统计用户各类交互事件的次数"""
    rows = fetch_all(
        "SELECT event_type, COUNT(*) as cnt FROM user_events WHERE user_id=%s GROUP BY event_type",
        (user_id,),
    )
    stats: dict = {}
    for etype, cnt in rows:
        stats[str(etype)] = int(cnt)
    return stats
