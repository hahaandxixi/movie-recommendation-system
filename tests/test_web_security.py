from myidea.views.movie import _is_safe_local_path


def test_redirect_target_must_be_local():
    assert _is_safe_local_path("/movie/1")
    assert not _is_safe_local_path("//example.com/path")
    assert not _is_safe_local_path("https://example.com/path")
