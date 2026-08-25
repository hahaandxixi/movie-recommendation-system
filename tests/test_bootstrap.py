import importlib
import sys

from bootstrap import ensure_source_package


def test_source_package_can_be_imported_when_clone_directory_has_another_name():
    alias = "myidea_clone_alias_test"
    ensure_source_package(alias)

    try:
        module = importlib.import_module(f"{alias}.core.data_loader")
        assert module.Movie.__name__ == "Movie"
    finally:
        for name in [key for key in sys.modules if key == alias or key.startswith(f"{alias}.")]:
            sys.modules.pop(name, None)
