from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, List

from myidea.config import ITEMCF_SIM_CACHE_PATH
from myidea.core.data_loader import Dataset
from myidea.core.recommender import (
    ContentBasedRecommender,
    ExplainedRecommendation,
    ItemCFRecommender,
    MostPopular,
    ScoredItem,
    TagBasedRecommender,
    UserCFRecommender,
    get_user_history,
)
from myidea.services.interaction_service import (
    get_user_interacted_items,
    get_user_profile_from_events,
    log_event,
    log_impressions,
)
from myidea.services.poster_service import get_poster_url, get_genre_colors

logger = logging.getLogger("myidea.services.recommendation_service")

_TWO_TOWER_MODEL = None


def _get_two_tower():
    """按需加载 TwoTower（双塔）模型。

为什么要“按需”：
- 双塔需要 numpy/pandas/torch，环境不全时可能会报错
- 课设的硬指标用不到双塔也能完成，所以这里做成“有就用、没有就跳过”
"""
    global _TWO_TOWER_MODEL
    if _TWO_TOWER_MODEL is None:
        try:
            from myidea.core.two_tower import TwoTowerRecommender, FeatureEncoder
            from pathlib import Path
            model_dir = Path(__file__).resolve().parents[1] / "models"
            _TWO_TOWER_MODEL = TwoTowerRecommender.from_saved(
                model_dir / "two_tower_model.pt",
                model_dir / "two_tower_encoder.pt",
            )
            _TWO_TOWER_MODEL.precompute_items()
            logger.info("TwoTower model loaded")
        except Exception as e:
            logger.warning("TwoTower model not available: %s", e)
            _TWO_TOWER_MODEL = False
    return _TWO_TOWER_MODEL if _TWO_TOWER_MODEL is not False else None


class RecommendationService:
    """推荐服务封装（把算法 + 日志 + 页面展示串起来）：
    - 热门榜：新用户/没历史时的兜底
    - ItemCF：主力个性化（也方便讲“为什么推荐”）
    - UserCF/Content/Tag：用于知识点覆盖、并排对比
    - TwoTower：深度学习示例（可选，有依赖就启用）
    """
    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self.popular = MostPopular(dataset)
        self._itemcf_cache_path = ITEMCF_SIM_CACHE_PATH
        self._itemcf_cache_autosave = os.environ.get("MYIDEA_ITEMCF_CACHE_AUTOSAVE", "1") == "1"
        self._itemcf_warmup = os.environ.get("MYIDEA_ITEMCF_WARMUP", "1") == "1"
        self._itemcf_fast_start = os.environ.get("MYIDEA_FAST_START", "0") == "1"

        cached_sim = self._load_itemcf_sim_cache()
        self.item_cf = ItemCFRecommender(
            dataset,
            sim=cached_sim,
            build_similarity=bool(cached_sim) and not self._itemcf_fast_start,
        )
        self.content_based = ContentBasedRecommender(dataset)
        self.user_cf = UserCFRecommender(dataset)
        self.tag_based = TagBasedRecommender(dataset)

    def warmup(self) -> None:
        """后台预热：避免用户第一次访问 /me 时遇到模型初始化带来的卡顿。"""
        if self._itemcf_fast_start or not self._itemcf_warmup:
            return
        if self._load_itemcf_sim_cache() is not None:
            return
        self.item_cf.ensure_ready()
        twotower = _get_two_tower()
        if twotower:
            twotower.precompute_items()
        self._save_itemcf_sim_cache()

    def _load_itemcf_sim_cache(self) -> Dict[int, Dict[int, float]] | None:
        """读取 ItemCF 相似度矩阵缓存（pickle），用于加速启动。"""
        try:
            p = self._itemcf_cache_path
            if not p.exists() or p.stat().st_size < 200:
                return None
            with p.open("rb") as f:
                sim = pickle.load(f)
            if isinstance(sim, dict):
                logger.info("ItemCF cache loaded: %s", p)
                return sim
        except Exception as exc:
            logger.warning("ItemCF cache load failed: %s", exc)
        return None

    def _save_itemcf_sim_cache(self) -> None:
        """保存 ItemCF 相似度矩阵缓存（pickle），避免下次启动重算。"""
        if not self._itemcf_cache_autosave:
            return
        try:
            p = self._itemcf_cache_path
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("wb") as f:
                pickle.dump(self.item_cf._sim, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("ItemCF cache saved: %s", p)
        except Exception as exc:
            logger.warning("ItemCF cache save failed: %s", exc)

    def get_popular(self, top_n: int = 12, exclude_user_id: int | None = None) -> List[Dict]:
        """热门榜：用于首页展示与冷启动兜底。"""
        exclude_items = set()
        if exclude_user_id is not None:
            for r in self._dataset.user_ratings.get(exclude_user_id, ()):
                exclude_items.add(r.movie_id)
            exclude_items |= get_user_interacted_items(exclude_user_id)

        items = self.popular.top_n(top_n, exclude_items=exclude_items)
        if exclude_user_id is not None:
            log_impressions(exclude_user_id, [it.item_id for it in items])
        out: List[Dict] = []
        for it in items:
            movie = self._dataset.movies.get(it.item_id)
            title = movie.title if movie else f"movie_id={it.item_id}"
            out.append({"movie_id": it.item_id, "title": title, "score": it.score})
        return [self._enrich(o) for o in out]

    def get_user_history(self, user_id: int, limit: int = 12) -> List[Dict]:
        history = get_user_history(self._dataset, user_id=user_id, limit=limit)
        out: List[Dict] = []
        for r in history:
            movie = self._dataset.movies.get(r.movie_id)
            title = movie.title if movie else f"movie_id={r.movie_id}"
            out.append({"movie_id": r.movie_id, "title": title, "rating": r.rating})
        return [self._enrich(o) for o in out]

    def get_recommendations(self, user_id: int, top_n: int = 12) -> List[Dict]:
        """个性化推荐（ItemCF）：融合历史评分与日志画像，并提供可解释推荐理由。"""
        profile: Dict[int, float] = {}

        # 画像来源 1：离线 MovieLens 历史评分（静态）
        base_history = get_user_history(self._dataset, user_id=user_id, limit=50)
        for r in base_history:
            profile[r.movie_id] = float(r.rating)

        # 画像来源 2：MySQL 在线交互日志（动态，按时间衰减）
        event_profile = get_user_profile_from_events(user_id=user_id, max_events=50)
        for item_id, weight in event_profile.items():
            profile[item_id] = max(profile.get(item_id, 0.0), weight)

        if not profile:
            return []

        recs: List[ExplainedRecommendation] = self.item_cf.recommend_from_profile(profile, n=top_n)
        log_impressions(user_id, [r.item_id for r in recs])
        self._save_itemcf_sim_cache()
        out: List[Dict] = []
        for r in recs:
            movie = self._dataset.movies.get(r.item_id)
            title = movie.title if movie else f"movie_id={r.item_id}"

            if r.reason_item_id is None:
                reason = "基于协同过滤的相似度扩散"
            else:
                src = self._dataset.movies.get(r.reason_item_id)
                src_title = src.title if src else f"movie_id={r.reason_item_id}"
                reason = f"因为你喜欢/评分过：{src_title}"

            out.append({"movie_id": r.item_id, "title": title, "score": r.score, "reason": reason})
        return [self._enrich(o) for o in out]

    def get_blended_recommendations(self, user_id: int, top_n: int = 20, seed: int = 0) -> List[Dict]:
        profile = self._build_profile(user_id)
        if not profile:
            fallback = self.get_popular(top_n=top_n, exclude_user_id=user_id)
            for it in fallback:
                it.setdefault("reason", "热门兜底")
                it.setdefault("algo", "popular")
            return fallback

        req_n = max(top_n * 6, top_n + 30)
        itemcf_raw = self.item_cf.recommend_from_profile(profile, n=req_n)
        content_raw = self.content_based.recommend_from_profile(profile, n=req_n)
        usercf_raw = self.user_cf.recommend(user_id, n=req_n)
        tag_raw = self.tag_based.recommend_from_profile(profile, n=req_n)

        def build_items(algo_key: str, recs: List[ExplainedRecommendation]) -> List[Dict]:
            items: List[Dict] = []
            for r in recs:
                movie = self._dataset.movies.get(r.item_id)
                title = movie.title if movie else f"movie_id={r.item_id}"
                if r.reason_item_id is not None:
                    src = self._dataset.movies.get(r.reason_item_id)
                    src_title = src.title if src else f"movie_id={r.reason_item_id}"
                    reason = f"因为你喜欢：{src_title}"
                else:
                    if algo_key == "usercf":
                        reason = "相似用户喜欢"
                    elif algo_key == "tag":
                        reason = "匹配你的偏好标签"
                    else:
                        reason = "基于相似度扩散"
                items.append({"movie_id": r.item_id, "title": title, "score": r.score, "reason": reason, "algo": algo_key})
            return items

        itemcf = build_items("itemcf", itemcf_raw)
        content = build_items("content", content_raw)
        usercf = build_items("usercf", usercf_raw)
        tag = build_items("tag", tag_raw)

        if seed:
            if itemcf:
                off = (seed + 11) % len(itemcf)
                itemcf = itemcf[off:] + itemcf[:off]
            if content:
                off = (seed + 23) % len(content)
                content = content[off:] + content[:off]
            if usercf:
                off = (seed + 37) % len(usercf)
                usercf = usercf[off:] + usercf[:off]
            if tag:
                off = (seed + 51) % len(tag)
                tag = tag[off:] + tag[:off]

        seen = set(profile.keys())
        out: List[Dict] = []
        buckets = [
            ("itemcf", itemcf),
            ("content", content),
            ("usercf", usercf),
            ("tag", tag),
        ]
        idxs = {k: 0 for k, _ in buckets}
        while len(out) < top_n and any(idxs[k] < len(lst) for k, lst in buckets):
            progressed = False
            for k, lst in buckets:
                idx = idxs[k]
                if idx >= len(lst):
                    continue
                idxs[k] = idx + 1
                cand = lst[idx]
                mid = int(cand["movie_id"])
                if mid in seen:
                    continue
                seen.add(mid)
                out.append(cand)
                progressed = True
                if len(out) >= top_n:
                    break
            if not progressed:
                break

        if len(out) < top_n:
            for it in self.get_popular(top_n=top_n, exclude_user_id=user_id):
                mid = int(it["movie_id"])
                if mid in seen:
                    continue
                it.setdefault("reason", "热门补位")
                it.setdefault("algo", "popular")
                out.append(it)
                if len(out) >= top_n:
                    break

        log_impressions(user_id, [int(x["movie_id"]) for x in out])
        self._save_itemcf_sim_cache()
        return [self._enrich(o) for o in out]

    def get_similar_items(self, item_id: int, top_n: int = 12) -> List[Dict]:
        out: List[Dict] = []
        for s in self.item_cf.similar_items(item_id=item_id, n=top_n):
            movie = self._dataset.movies.get(s.item_id)
            title = movie.title if movie else f"movie_id={s.item_id}"
            out.append({"movie_id": s.item_id, "title": title, "score": s.score, "algo": "itemcf"})
        self._save_itemcf_sim_cache()
        for s in self.content_based.similar_items(item_id=item_id, n=top_n):
            movie = self._dataset.movies.get(s.item_id)
            title = movie.title if movie else f"movie_id={s.item_id}"
            out.append({"movie_id": s.item_id, "title": title, "score": s.score, "algo": "content"})
        return [self._enrich(o) for o in out]

    def get_multi_recommendations(
        self,
        user_id: int,
        top_n: int = 8,
        refresh_algo: str = "",
        refresh_seed: int = 0,
    ) -> Dict[str, List[Dict]]:
        """返回多算法推荐结果，供页面并排对比展示"""
        profile = self._build_profile(user_id)
        if not profile:
            # 冷启动：用户没有历史评分/交互时，用热门榜兜底，保证页面始终有结果
            fallback = self.get_popular(top_n=top_n, exclude_user_id=user_id)
            for it in fallback:
                it.setdefault("reason", "")
            return {
                "itemcf": fallback,
                "content": fallback,
                "usercf": fallback,
                "tag": fallback,
            }

        result: Dict[str, List[Dict]] = {}
        algo_list = [
            ("itemcf", self.item_cf),
            ("content", self.content_based),
            ("usercf", self.user_cf),
            ("tag", self.tag_based),
        ]

        twotower = _get_two_tower()
        if twotower:
            algo_list.append(("twotower", None))

        for algo_name, algo in algo_list:
            req_n = top_n
            if algo_name == refresh_algo and refresh_seed:
                req_n = max(top_n * 4, top_n + 10)
            if algo_name == "twotower":
                skipped = set(profile.keys())
                recs_raw = twotower.recommend(user_id, top_n=req_n, exclude_items=skipped)
                log_impressions(user_id, [mid for mid, _ in recs_raw])
                items: List[Dict] = []
                for mid, score in recs_raw:
                    movie = self._dataset.movies.get(mid)
                    title = movie.title if movie else f"movie_id={mid}"
                    items.append({"movie_id": mid, "title": title, "score": score, "reason": "双塔模型计算"})
                if algo_name == refresh_algo and refresh_seed and len(items) > top_n:
                    offset = refresh_seed % max(1, len(items) - top_n + 1)
                    items = items[offset : offset + top_n]
                else:
                    items = items[:top_n]
                result[algo_name] = [self._enrich(o) for o in items]
            else:
                recs = algo.recommend_from_profile(profile, n=req_n)
                log_impressions(user_id, [r.item_id for r in recs])
                items = []
                for r in recs:
                    movie = self._dataset.movies.get(r.item_id)
                    title = movie.title if movie else f"movie_id={r.item_id}"
                    if r.reason_item_id is not None:
                        src = self._dataset.movies.get(r.reason_item_id)
                        src_title = src.title if src else f"movie_id={r.reason_item_id}"
                        reason = f"因为你喜欢：{src_title}"
                    else:
                        reason = f"{algo_name} 计算"
                    items.append({"movie_id": r.item_id, "title": title, "score": r.score, "reason": reason})
                if algo_name == refresh_algo and refresh_seed and len(items) > top_n:
                    offset = refresh_seed % max(1, len(items) - top_n + 1)
                    items = items[offset : offset + top_n]
                else:
                    items = items[:top_n]
                result[algo_name] = [self._enrich(o) for o in items]

        return result

    def _build_profile(self, user_id: int) -> Dict[int, float]:
        profile: Dict[int, float] = {}
        base_history = get_user_history(self._dataset, user_id=user_id, limit=50)
        for r in base_history:
            profile[r.movie_id] = float(r.rating)
        event_profile = get_user_profile_from_events(user_id=user_id, max_events=50)
        for item_id, weight in event_profile.items():
            profile[item_id] = max(profile.get(item_id, 0.0), weight)
        return profile

    def get_movie_info(self, item_id: int) -> Dict | None:
        movie = self._dataset.movies.get(item_id)
        if movie is None:
            return None
        genres_str = " | ".join(movie.genres) if movie.genres else "无分类"
        poster_url = get_poster_url(movie.movie_id, movie.title, genres_str)
        info = {
            "movie_id": movie.movie_id,
            "title": movie.title,
            "genres": genres_str,
            "poster_url": poster_url,
        }
        if not poster_url:
            c1, c2 = get_genre_colors(genres_str)
            info["poster_c1"] = c1
            info["poster_c2"] = c2
            info["poster_label"] = genres_str.split(" | ")[0] if genres_str else "Film"
        return info

    def _enrich(self, item: Dict) -> Dict:
        mid = item.get("movie_id")
        if mid is None:
            return item
        movie = self._dataset.movies.get(mid)
        if movie is None:
            return item
        genres_str = " | ".join(movie.genres) if movie.genres else ""
        poster_url = get_poster_url(movie.movie_id, movie.title, genres_str)
        item.setdefault("poster_url", poster_url)
        item.setdefault("genres", genres_str)
        if not poster_url:
            c1, c2 = get_genre_colors(genres_str)
            item.setdefault("poster_c1", c1)
            item.setdefault("poster_c2", c2)
            primary = genres_str.split(" | ")[0] if genres_str else "Film"
            item.setdefault("poster_label", primary)
        return item
