"""双塔模型推荐算法
基于 MovieLens 1M 用户/物品特征训练双塔神经网络进行评分预测

架构：
- User Tower：user_id 嵌入(32) + gender + age 嵌入(16) + occupation 嵌入(16) → MLP(64→32)
- Item Tower：movie_id 嵌入(32) + genres 多热(18) → MLP(64→32)
- 相似度计算：L2 归一化后内积 → sigmoid → 线性缩放至 1-5

特征来源：
- users.dat：gender(M/F), age(1/18/25/35/45/50/56), occupation(0-20)
- movies.dat：genres(18种类型多热编码)
- ratings.dat：rating(1-5) 作为训练标签

训练：MSE 损失 + Adam 优化器 + 5 epoch，测试 MAE ≈ 0.76
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("myidea.core.two_tower")

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
DATASET_DIR = Path(__file__).resolve().parents[1] / "数据集" / "MovieLens" / "ml-1m" / "ml-1m"


class FeatureEncoder:
    """特征编码器：将 MovieLens 原始 ID 映射为模型可用的数值特征
    负责 user_id/movie_id 到连续索引的映射、genre 多热编码、年龄/职业离散化"""

    def __init__(self):
        self.user_id_to_idx: Dict[int, int] = {}
        self.movie_id_to_idx: Dict[int, int] = {}
        self.idx_to_user_id: Dict[int, int] = {}
        self.idx_to_movie_id: Dict[int, int] = {}
        self.num_users: int = 0
        self.num_movies: int = 0
        self.num_occupations: int = 0
        self.num_ages: int = 0
        self.genre_names: List[str] = []
        self.movie_genres: Dict[int, np.ndarray] = {}

    def fit(self, ratings_df, users_df, movies_df):
        user_ids = ratings_df["user_id"].unique()
        movie_ids = ratings_df["movie_id"].unique()
        for idx, uid in enumerate(user_ids):
            self.user_id_to_idx[uid] = idx
            self.idx_to_user_id[idx] = uid
        for idx, mid in enumerate(movie_ids):
            self.movie_id_to_idx[mid] = idx
            self.idx_to_movie_id[idx] = mid
        self.num_users = len(user_ids)
        self.num_movies = len(movie_ids)

        self.num_occupations = users_df["occupation"].nunique()
        self.num_ages = users_df["age"].nunique()
        self.age_to_idx = {a: i for i, a in enumerate(sorted(users_df["age"].unique()))}
        self.occ_to_idx = {o: i for i, o in enumerate(sorted(users_df["occupation"].unique()))}

        all_genres = set()
        for gs in movies_df["genres"].dropna():
            for g in gs.split("|"):
                all_genres.add(g.strip())
        self.genre_names = sorted(all_genres)

        for _, row in movies_df.iterrows():
            mid = row["movie_id"]
            vec = np.zeros(len(self.genre_names), dtype=np.float32)
            if pd.notna(row["genres"]):
                for g in row["genres"].split("|"):
                    g = g.strip()
                    if g in self.genre_names:
                        vec[self.genre_names.index(g)] = 1.0
            self.movie_genres[mid] = vec

    def encode_sample(self, row) -> dict:
        uid = row["user_id"]
        mid = row["movie_id"]
        return {
            "user_idx": torch.tensor(self.user_id_to_idx.get(uid, 0), dtype=torch.long),
            "movie_idx": torch.tensor(self.movie_id_to_idx.get(mid, 0), dtype=torch.long),
            "gender": torch.tensor(1.0 if row.get("gender") == "M" else 0.0, dtype=torch.float),
            "age_idx": torch.tensor(self.age_to_idx.get(row.get("age", 25), 0), dtype=torch.long),
            "occ_idx": torch.tensor(self.occ_to_idx.get(row.get("occupation", 0), 0), dtype=torch.long),
            "genres": torch.from_numpy(self.movie_genres.get(mid, np.zeros(len(self.genre_names), dtype=np.float32))),
        }

    def encode_batch(self, df) -> dict:
        users = torch.tensor([self.user_id_to_idx.get(uid, 0) for uid in df["user_id"]], dtype=torch.long)
        movies = torch.tensor([self.movie_id_to_idx.get(mid, 0) for mid in df["movie_id"]], dtype=torch.long)
        genders = torch.tensor([1.0 if g == "M" else 0.0 for g in df["gender"]], dtype=torch.float)
        ages = torch.tensor([self.age_to_idx.get(a, 0) for a in df["age"]], dtype=torch.long)
        occs = torch.tensor([self.occ_to_idx.get(o, 0) for o in df["occupation"]], dtype=torch.long)
        genre_vecs = torch.stack([torch.from_numpy(self.movie_genres.get(mid, np.zeros(len(self.genre_names), dtype=np.float32))) for mid in df["movie_id"]])
        return {"user_idx": users, "movie_idx": movies, "gender": genders, "age_idx": ages, "occ_idx": occs, "genres": genre_vecs}

    def save(self, path):
        torch.save({
            "user_id_to_idx": self.user_id_to_idx,
            "movie_id_to_idx": self.movie_id_to_idx,
            "idx_to_user_id": self.idx_to_user_id,
            "idx_to_movie_id": self.idx_to_movie_id,
            "num_users": self.num_users,
            "num_movies": self.num_movies,
            "num_occupations": self.num_occupations,
            "num_ages": self.num_ages,
            "age_to_idx": self.age_to_idx,
            "occ_to_idx": self.occ_to_idx,
            "genre_names": self.genre_names,
            "movie_genres": self.movie_genres,
        }, path)

    def load(self, path):
        data = torch.load(path, weights_only=False)
        for k, v in data.items():
            setattr(self, k, v)
        return self


class TwoTowerModel(nn.Module):
    """双塔神经网络模型
    User Tower 输入：user_id 嵌入 + gender 标量 + age 嵌入 + occupation 嵌入
    Item Tower 输入：movie_id 嵌入 + genres 多热向量
    输出：L2 归一化后再点积，经 sigmoid 缩放为 1-5 评分"""

    def __init__(self, num_users, num_movies, num_ages, num_occs, num_genres, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.user_id_emb = nn.Embedding(num_users, embed_dim)
        self.movie_id_emb = nn.Embedding(num_movies, embed_dim)
        self.age_emb = nn.Embedding(num_ages, embed_dim // 2)
        self.occ_emb = nn.Embedding(num_occs, embed_dim // 2)

        user_input_dim = embed_dim + 1 + embed_dim // 2 + embed_dim // 2
        self.user_tower = nn.Sequential(
            nn.Linear(user_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        item_input_dim = embed_dim + num_genres
        self.item_tower = nn.Sequential(
            nn.Linear(item_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.user_id_emb, self.movie_id_emb, self.age_emb, self.occ_emb]:
            nn.init.normal_(m.weight, std=0.01)
        for module in [self.user_tower, self.item_tower]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, batch):
        user_vec = self.user_embeddings(batch)
        item_vec = self.item_embeddings(batch)
        dot = (user_vec * item_vec).sum(dim=1)
        return torch.sigmoid(dot) * 4.0 + 1.0

    def user_embeddings(self, batch):
        uid_emb = self.user_id_emb(batch["user_idx"])
        age_emb = self.age_emb(batch["age_idx"])
        occ_emb = self.occ_emb(batch["occ_idx"])
        gender = batch["gender"].unsqueeze(1)
        combined = torch.cat([uid_emb, gender, age_emb, occ_emb], dim=1)
        return F.normalize(self.user_tower(combined), dim=1)

    def item_embeddings(self, batch):
        mid_emb = self.movie_id_emb(batch["movie_idx"])
        combined = torch.cat([mid_emb, batch["genres"]], dim=1)
        return F.normalize(self.item_tower(combined), dim=1)


class TwoTowerRecommender:
    """双塔推荐推理器
    加载训练好的模型和编码器，在线推理时：
    1. 预计算全部物品向量（离线一次性）
    2. 对目标用户实时编码 → 与全部物品向量内积 → TopN"""

    def __init__(self, encoder: FeatureEncoder, model: TwoTowerModel, device="cpu"):
        self.encoder = encoder
        self.model = model.to(device).eval()
        self.device = device
        self._user_cache: Dict[int, torch.Tensor] = {}
        self._item_embs: Optional[torch.Tensor] = None
        self._item_ids: List[int] = []

    def precompute_items(self):
        if self._item_embs is not None:
            return
        genre_dim = len(self.encoder.genre_names)
        item_list = sorted(self.encoder.movie_id_to_idx.keys())
        self._item_ids = item_list
        genre_batch = np.zeros((len(item_list), genre_dim), dtype=np.float32)
        movie_batch = np.zeros(len(item_list), dtype=np.int64)
        for i, mid in enumerate(item_list):
            idx = self.encoder.movie_id_to_idx.get(mid, 0)
            movie_batch[i] = idx
            genre_batch[i] = self.encoder.movie_genres.get(mid, np.zeros(genre_dim, dtype=np.float32))
        batch = {
            "movie_idx": torch.tensor(movie_batch).to(self.device),
            "genres": torch.tensor(genre_batch).to(self.device),
        }
        with torch.no_grad():
            self._item_embs = self.model.item_embeddings(batch)

    def get_user_embedding(self, user_id: int, gender="M", age=25, occupation=0):
        if user_id not in self.encoder.user_id_to_idx:
            return None
        user_idx = self.encoder.user_id_to_idx[user_id]
        age_idx = self.encoder.age_to_idx.get(age, 0)
        occ_idx = self.encoder.occ_to_idx.get(occupation, 0)
        batch = {
            "user_idx": torch.tensor([user_idx]).to(self.device),
            "gender": torch.tensor([1.0 if gender == "M" else 0.0]).to(self.device),
            "age_idx": torch.tensor([age_idx]).to(self.device),
            "occ_idx": torch.tensor([occ_idx]).to(self.device),
        }
        with torch.no_grad():
            return self.model.user_embeddings(batch).cpu()

    def recommend(self, user_id: int, top_n: int = 20, exclude_items=None) -> List[Tuple[int, float]]:
        self.precompute_items()
        user_emb = self.get_user_embedding(user_id)
        if user_emb is None:
            return []
        scores = torch.matmul(user_emb, self._item_embs.T).squeeze(0)
        exclude_set = set(exclude_items or [])
        scored = []
        for i, mid in enumerate(self._item_ids):
            if mid in exclude_set:
                continue
            scored.append((mid, float(scores[i])))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    @classmethod
    def from_saved(cls, model_path, encoder_path, device="cpu"):
        encoder = FeatureEncoder().load(encoder_path)
        model = TwoTowerModel(
            encoder.num_users, encoder.num_movies,
            encoder.num_ages, encoder.num_occupations,
            len(encoder.genre_names),
            embed_dim=32, hidden_dim=64,
        )
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        return cls(encoder, model, device)


def load_movielens_data(data_dir):
    ratings = pd.read_csv(data_dir / "ratings.dat", sep="::", engine="python",
                          names=["user_id", "movie_id", "rating", "timestamp"])
    users = pd.read_csv(data_dir / "users.dat", sep="::", engine="python",
                         names=["user_id", "gender", "age", "occupation", "zip"])
    movies = pd.read_csv(data_dir / "movies.dat", sep="::", engine="python",
                          names=["movie_id", "title", "genres"], encoding="latin-1")
    return ratings, users, movies


def train_two_tower(epochs=5, batch_size=512, lr=0.001, device="cpu"):
    ratings, users, movies = load_movielens_data(DATASET_DIR)

    merged = ratings.merge(users, on="user_id").merge(movies, on="movie_id")

    encoder = FeatureEncoder()
    encoder.fit(ratings, users, movies)

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(merged, test_size=0.2, random_state=42)

    model = TwoTowerModel(
        encoder.num_users, encoder.num_movies,
        encoder.num_ages, encoder.num_occupations,
        len(encoder.genre_names),
        embed_dim=32, hidden_dim=64,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        perm = torch.randperm(len(train_df))
        for i in range(0, len(train_df), batch_size):
            idxs = perm[i:i + batch_size]
            batch = encoder.encode_batch(train_df.iloc[idxs.numpy()])
            batch = {k: v.to(device) for k, v in batch.items()}
            targets = torch.tensor(train_df.iloc[idxs.numpy()]["rating"].values, dtype=torch.float).to(device)

            preds = model(batch)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idxs)

        model.eval()
        test_batch = encoder.encode_batch(test_df)
        test_batch_gpu = {k: v.to(device) for k, v in test_batch.items()}
        test_targets = torch.tensor(test_df["rating"].values, dtype=torch.float).to(device)
        with torch.no_grad():
            test_preds = model(test_batch_gpu)
            test_loss = criterion(test_preds, test_targets).item()
            test_mae = F.l1_loss(test_preds, test_targets).item()

        logger.info("Epoch %d/%d | train_loss=%.4f | test_loss=%.4f | test_mae=%.4f",
                     epoch + 1, epochs, total_loss / len(train_df), test_loss, test_mae)

    MODEL_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "two_tower_model.pt")
    encoder.save(MODEL_DIR / "two_tower_encoder.pt")
    logger.info("Model saved to %s", MODEL_DIR)
    return model, encoder
