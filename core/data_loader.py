from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True, slots=True)
class Movie:
    movie_id: int
    title: str
    genres: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Rating:
    user_id: int
    movie_id: int
    rating: float
    timestamp: int


@dataclass(frozen=True, slots=True)
class Dataset:
    movies: Mapping[int, Movie]
    ratings: Tuple[Rating, ...]
    user_ratings: Mapping[int, Tuple[Rating, ...]]


def load_movielens_dat(dataset_dir: str | Path) -> Dataset:
    dataset_path = Path(dataset_dir)
    movies_path = dataset_path / "movies.dat"
    ratings_path = dataset_path / "ratings.dat"

    movies = _load_movies_dat(movies_path)
    ratings = tuple(_load_ratings_dat(ratings_path))

    user_ratings: Dict[int, List[Rating]] = {}
    for r in ratings:
        user_ratings.setdefault(r.user_id, []).append(r)

    user_ratings_sorted: Dict[int, Tuple[Rating, ...]] = {}
    for user_id, rs in user_ratings.items():
        rs.sort(key=lambda x: x.timestamp, reverse=True)
        user_ratings_sorted[user_id] = tuple(rs)

    return Dataset(movies=movies, ratings=ratings, user_ratings=user_ratings_sorted)


def _load_movies_dat(path: Path) -> Dict[int, Movie]:
    movies: Dict[int, Movie] = {}
    for line in _read_lines(path):
        raw_movie_id, raw_title, raw_genres = _split_ml_dat_line(line, expected_parts=3)
        movie_id = int(raw_movie_id)
        title = raw_title.strip()
        genres = tuple(g for g in raw_genres.split("|") if g)
        movies[movie_id] = Movie(movie_id=movie_id, title=title, genres=genres)
    return movies


def _load_ratings_dat(path: Path) -> Iterable[Rating]:
    for line in _read_lines(path):
        raw_user_id, raw_movie_id, raw_rating, raw_ts = _split_ml_dat_line(line, expected_parts=4)
        yield Rating(
            user_id=int(raw_user_id),
            movie_id=int(raw_movie_id),
            rating=float(raw_rating),
            timestamp=int(raw_ts),
        )


def _read_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                yield s


def _split_ml_dat_line(line: str, expected_parts: int) -> List[str]:
    parts = line.split("::")
    if len(parts) != expected_parts:
        raise ValueError(f"Invalid .dat line, expected {expected_parts} parts, got {len(parts)}: {line!r}")
    return parts

