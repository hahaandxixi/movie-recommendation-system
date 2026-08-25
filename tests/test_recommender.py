from myidea.core.data_loader import Dataset, Movie, Rating
from myidea.core.recommender import ContentBasedRecommender, MostPopular


def _dataset() -> Dataset:
    movies = {
        1: Movie(1, "Action One", ("Action",)),
        2: Movie(2, "Action Two", ("Action",)),
        3: Movie(3, "Comedy", ("Comedy",)),
    }
    ratings = (
        Rating(1, 1, 5.0, 1),
        Rating(2, 1, 4.0, 2),
        Rating(2, 2, 5.0, 3),
        Rating(3, 3, 3.0, 4),
    )
    user_ratings = {
        1: (ratings[0],),
        2: (ratings[2], ratings[1]),
        3: (ratings[3],),
    }
    return Dataset(movies=movies, ratings=ratings, user_ratings=user_ratings)


def test_popular_recommendations_respect_exclusions():
    result = MostPopular(_dataset()).top_n(3, exclude_items={1})

    assert all(item.item_id != 1 for item in result)


def test_content_recommendations_exclude_seen_movies():
    result = ContentBasedRecommender(_dataset()).recommend_from_profile({1: 5.0}, n=3)

    assert [item.item_id for item in result] == [2]
