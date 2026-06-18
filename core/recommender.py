"""推荐算法核心模块
实现 5 种推荐算法：
- MostPopular: 热门榜（按评分次数+均值加权排序，用于冷启动兜底）
- ItemCFRecommender: 基于物品的协同过滤（物品共现余弦相似度）
- ContentBasedRecommender: 基于物品内容的推荐（genres Jaccard 相似度 + 用户偏好聚合）
- UserCFRecommender: 基于用户的协同过滤（用户余弦相似度 + TopK 邻居加权）
- TagBasedRecommender: 基于标签的推荐（genres 作标签，统计用户标签偏好排序）
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Set, Tuple

from myidea.core.data_loader import Dataset, Rating

logger = logging.getLogger("myidea.core.recommender")


@dataclass(frozen=True, slots=True)
class ScoredItem:
    """通用评分结构：物品 ID + 分数"""
    item_id: int
    score: float


@dataclass(frozen=True, slots=True)
class ExplainedRecommendation:
    """带解释的推荐结果：包含推荐源物品和解释强度"""
    item_id: int
    score: float
    reason_item_id: int | None
    reason_strength: float | None


class MostPopular:
    """热门榜推荐算法
    计算口径：评分次数 * 0.7 + 平均评分 * 0.3
    用于冷启动用户（无历史记录）和兜底推荐"""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._counts: Dict[int, int] = {}
        self._sum_ratings: Dict[int, float] = {}
        for r in dataset.ratings:
            self._counts[r.movie_id] = self._counts.get(r.movie_id, 0) + 1
            self._sum_ratings[r.movie_id] = self._sum_ratings.get(r.movie_id, 0.0) + r.rating

    def top_n(self, n: int, exclude_items: Set[int] | None = None) -> List[ScoredItem]:
        exclude_items = exclude_items or set()
        scored: List[Tuple[float, int]] = []
        for movie_id, cnt in self._counts.items():
            if movie_id in exclude_items:
                continue
            avg = self._sum_ratings[movie_id] / cnt
            score = cnt * 0.7 + avg * 0.3
            scored.append((score, movie_id))
        scored.sort(reverse=True)
        return [ScoredItem(item_id=movie_id, score=score) for score, movie_id in scored[:n]]


class ItemCFRecommender:
    """基于物品的协同过滤推荐算法
    原理：统计物品在用户行为中的共现次数，计算余弦相似度
    公式：sim(a,b) = co_count(a,b) / sqrt(|users(a)| * |users(b)|)
    推荐：用户历史物品 * 相似度矩阵 → TopN"""

    def __init__(
        self,
        dataset: Dataset,
        topk_sim_per_item: int = 50,
        sim: Dict[int, Dict[int, float]] | None = None,
        build_similarity: bool = True,
    ) -> None:
        self._dataset = dataset
        self._topk_sim_per_item = topk_sim_per_item
        self._item_users: Dict[int, Set[int]] = {}
        self._user_items: Dict[int, Dict[int, float]] = {}

        for r in dataset.ratings:
            self._item_users.setdefault(r.movie_id, set()).add(r.user_id)
            self._user_items.setdefault(r.user_id, {})[r.movie_id] = r.rating

        if sim is not None:
            self._sim: Dict[int, Dict[int, float]] | None = sim
        elif build_similarity:
            self._sim = self._build_similarity()
        else:
            self._sim = None

    def ensure_ready(self) -> None:
        self._ensure_sim()

    def _ensure_sim(self) -> None:
        if self._sim is None:
            self._sim = self._build_similarity()

    def recommend(self, user_id: int, n: int) -> List[ExplainedRecommendation]:
        user_item_ratings = self._user_items.get(user_id)
        if not user_item_ratings:
            return []
        return self.recommend_from_profile(user_item_ratings, n=n)

    def recommend_from_profile(self, user_item_ratings: Mapping[int, float], n: int) -> List[ExplainedRecommendation]:
        self._ensure_sim()
        seen = set(user_item_ratings.keys())
        scores: Dict[int, float] = {}
        reasons: Dict[int, Tuple[int, float]] = {}

        for src_item_id, rating in user_item_ratings.items():
            neighbors = self._sim.get(src_item_id, {}) if self._sim else {}
            for dst_item_id, sim in neighbors.items():
                if dst_item_id in seen:
                    continue
                weight = sim * rating
                scores[dst_item_id] = scores.get(dst_item_id, 0.0) + weight
                best = reasons.get(dst_item_id)
                if best is None or weight > best[1]:
                    reasons[dst_item_id] = (src_item_id, weight)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        recs: List[ExplainedRecommendation] = []
        for item_id, score in ranked:
            reason = reasons.get(item_id)
            if reason:
                recs.append(
                    ExplainedRecommendation(
                        item_id=item_id,
                        score=score,
                        reason_item_id=reason[0],
                        reason_strength=reason[1],
                    )
                )
            else:
                recs.append(
                    ExplainedRecommendation(
                        item_id=item_id,
                        score=score,
                        reason_item_id=None,
                        reason_strength=None,
                    )
                )
        return recs

    def similar_items(self, item_id: int, n: int) -> List[ScoredItem]:
        self._ensure_sim()
        neighbors = self._sim.get(item_id, {}) if self._sim else {}
        scored = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)[:n]
        return [ScoredItem(item_id=dst_item_id, score=score) for dst_item_id, score in scored]

    def _build_similarity(self) -> Dict[int, Dict[int, float]]:
        item_ids = list(self._item_users.keys())
        item_count = {item_id: len(users) for item_id, users in self._item_users.items()}

        co_count: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        user_count = len(self._user_items)
        report_every = max(1, user_count // 10)

        for idx, (user_id, item_ratings) in enumerate(self._user_items.items()):
            items = list(item_ratings.keys())
            n = len(items)
            for i in range(n):
                a = items[i]
                a_neighbors = co_count[a]
                for j in range(i + 1, n):
                    b = items[j]
                    a_neighbors[b] += 1
                    co_count[b][a] += 1
            if (idx + 1) % report_every == 0:
                logger.info("ItemCF similarity: %d/%d users processed", idx + 1, user_count)

        logger.info("ItemCF similarity: co-occurrence done, computing cosine scores")
        sim: Dict[int, Dict[int, float]] = {}
        for a in item_ids:
            neighbors = co_count.get(a)
            if not neighbors:
                continue
            a_cnt = item_count.get(a, 0)
            if a_cnt == 0:
                continue
            scored: List[Tuple[float, int]] = []
            for b, c in neighbors.items():
                b_cnt = item_count.get(b, 0)
                if b_cnt == 0:
                    continue
                s = c / math.sqrt(a_cnt * b_cnt)
                if s > 0.0:
                    scored.append((s, b))
            scored.sort(reverse=True)
            top = scored[: self._topk_sim_per_item]
            sim[a] = {b: s for s, b in top}
        logger.info("ItemCF similarity: done, %d items in matrix", len(sim))
        return sim


class ContentBasedRecommender:
    """基于物品内容的推荐算法
    原理：电影 genres 做 Jaccard 相似度，聚合用户偏好类型权重
    公式：推荐分 = sum(genre_weight[g] for g in movie.genres)
    解释：找历史中与推荐物品类型最相似的作品作为推荐理由"""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._movie_genres: Dict[int, Set[str]] = {}
        for movie in dataset.movies.values():
            self._movie_genres[movie.movie_id] = set(movie.genres)

    def similar_items(self, item_id: int, n: int) -> List[ScoredItem]:
        src_genres = self._movie_genres.get(item_id)
        if not src_genres:
            return []

        scored: List[Tuple[float, int]] = []
        for mid, genres in self._movie_genres.items():
            if mid == item_id:
                continue
            intersection = len(src_genres & genres)
            union = len(src_genres | genres)
            if union == 0:
                continue
            jaccard = intersection / union
            if jaccard > 0.0:
                scored.append((jaccard, mid))
        scored.sort(reverse=True)
        return [ScoredItem(item_id=mid, score=s) for s, mid in scored[:n]]

    def recommend_from_profile(self, user_item_ratings: Mapping[int, float], n: int) -> List[ExplainedRecommendation]:
        seen = set(user_item_ratings.keys())

        # 聚合用户偏好类型
        genre_weights: Dict[str, float] = defaultdict(float)
        for item_id, rating in user_item_ratings.items():
            genres = self._movie_genres.get(item_id, set())
            for g in genres:
                genre_weights[g] += rating

        if not genre_weights:
            return []

        # 计算未看物品与用户偏好类型的匹配度
        scored: Dict[int, float] = {}
        reasons: Dict[int, Tuple[int, float]] = {}
        for movie in self._dataset.movies.values():
            if movie.movie_id in seen:
                continue
            mg = self._movie_genres.get(movie.movie_id, set())
            if not mg:
                continue
            score = sum(genre_weights.get(g, 0.0) for g in mg)
            if score <= 0.0:
                continue
            scored[movie.movie_id] = score
            # 找最匹配的历史物品作解释
            best_sim = 0.0
            best_src = None
            for src_id in user_item_ratings:
                sg = self._movie_genres.get(src_id, set())
                inter = len(mg & sg)
                uni = len(mg | sg)
                sim = inter / uni if uni > 0 else 0.0
                if sim > best_sim:
                    best_sim = sim
                    best_src = src_id
            reasons[movie.movie_id] = (best_src or 0, best_sim)

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:n]
        return [
            ExplainedRecommendation(
                item_id=mid, score=s,
                reason_item_id=reasons[mid][0] if reasons[mid][0] != 0 else None,
                reason_strength=reasons[mid][1],
            )
            for mid, s in ranked
        ]


class UserCFRecommender:
    """基于用户的协同过滤推荐算法
    原理：计算用户间余弦相似度，Top50 邻居的加权评分作为推荐依据
    公式：sim(u,v) = cosine(normalized_ratings(u), normalized_ratings(v))
    推荐：sum(sim(u,neighbor) * rating(neighbor,item) for neighbor in topK)"""

    def __init__(self, dataset: Dataset, topk_users: int = 50) -> None:
        self._dataset = dataset
        self._topk_users = topk_users
        self._user_ids = list(dataset.user_ratings.keys())
        self._user_ratings: Dict[int, Dict[int, float]] = {}
        for uid, ratings in dataset.user_ratings.items():
            self._user_ratings[uid] = {r.movie_id: r.rating for r in ratings}

    def recommend(self, user_id: int, n: int) -> List[ExplainedRecommendation]:
        if user_id not in self._user_ratings:
            return []

        target = self._user_ratings[user_id]
        seen = set(target.keys())

        # 计算目标用户与所有用户的余弦相似度
        sims: List[Tuple[float, int]] = []
        target_vec = _norm_vec(target)
        for uid in self._user_ids:
            if uid == user_id:
                continue
            other = self._user_ratings[uid]
            sim = _cosine(target_vec, _norm_vec(other))
            if sim > 0.0:
                sims.append((sim, uid))
        sims.sort(reverse=True)

        # top-k 相似用户加权
        scores: Dict[int, float] = {}
        reasons: Dict[int, Tuple[int, float]] = {}
        for sim, uid in sims[: self._topk_users]:
            for mid, rating in self._user_ratings[uid].items():
                if mid in seen:
                    continue
                w = sim * rating
                scores[mid] = scores.get(mid, 0.0) + w
                best = reasons.get(mid)
                if best is None or w > best[1]:
                    reasons[mid] = (uid, w)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
        return [
            ExplainedRecommendation(
                item_id=mid, score=s,
                reason_item_id=None,
                reason_strength=reasons[mid][1],
            )
            for mid, s in ranked
        ]

    def recommend_from_profile(self, user_item_ratings: Mapping[int, float], n: int) -> List[ExplainedRecommendation]:
        return self.recommend(list(user_item_ratings.keys())[0] if user_item_ratings else 0, n)


class TagBasedRecommender:
    """基于标签的推荐算法
    原理：把电影 genres 作为标签，统计用户历史中每个标签的累计偏好权重
    推荐：对未看物品按匹配到的标签权重之和排序，标签匹配越多分数越高"""

    def __init__(self, dataset: Dataset) -> None:
        self._dataset = dataset
        self._movie_genres: Dict[int, Set[str]] = {}
        for movie in dataset.movies.values():
            self._movie_genres[movie.movie_id] = set(movie.genres)

    def recommend_from_profile(self, user_item_ratings: Mapping[int, float], n: int) -> List[ExplainedRecommendation]:
        seen = set(user_item_ratings.keys())

        tag_weights: Dict[str, float] = defaultdict(float)
        for item_id, rating in user_item_ratings.items():
            genres = self._movie_genres.get(item_id, set())
            for g in genres:
                tag_weights[g] += rating

        if not tag_weights:
            return []

        # 找标签与用户偏好匹配的电影，按 sum(tag_weight * match) 排序
        scored: Dict[int, float] = {}
        for movie in self._dataset.movies.values():
            if movie.movie_id in seen:
                continue
            mg = self._movie_genres.get(movie.movie_id, set())
            score = sum(tag_weights.get(g, 0.0) for g in mg)
            if score > 0.0:
                scored[movie.movie_id] = score

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)[:n]
        return [
            ExplainedRecommendation(
                item_id=mid, score=s,
                reason_item_id=None, reason_strength=None,
            )
            for mid, s in ranked
        ]

    def similar_items(self, item_id: int, n: int) -> List[ScoredItem]:
        src_genres = self._movie_genres.get(item_id)
        if not src_genres:
            return []
        scored: List[Tuple[float, int]] = []
        for mid, genres in self._movie_genres.items():
            if mid == item_id:
                continue
            overlap = len(src_genres & genres)
            if overlap > 0:
                scored.append((float(overlap), mid))
        scored.sort(reverse=True)
        return [ScoredItem(item_id=mid, score=s) for s, mid in scored[:n]]


def _norm_vec(ratings: Dict[int, float]) -> Dict[int, float]:
    norm = math.sqrt(sum(v * v for v in ratings.values()))
    if norm == 0:
        return ratings
    return {k: v / norm for k, v in ratings.items()}


def _cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(a[k] * b[k] for k in a if k in b)


def get_user_history(dataset: Dataset, user_id: int, limit: int = 20) -> Sequence[Rating]:
    ratings = dataset.user_ratings.get(user_id, ())
    return ratings[:limit]
