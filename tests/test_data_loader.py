from myidea.core.data_loader import load_movielens_dat


def test_load_movielens_dat_builds_movies_and_user_history(tmp_path):
    (tmp_path / "movies.dat").write_text(
        "1::Toy Story (1995)::Animation|Comedy\n"
        "2::Heat (1995)::Action|Crime\n",
        encoding="utf-8",
    )
    (tmp_path / "ratings.dat").write_text(
        "7::1::4::100\n"
        "7::2::5::200\n",
        encoding="utf-8",
    )

    dataset = load_movielens_dat(tmp_path)

    assert dataset.movies[1].title == "Toy Story (1995)"
    assert dataset.movies[2].genres == ("Action", "Crime")
    assert [rating.movie_id for rating in dataset.user_ratings[7]] == [2, 1]
