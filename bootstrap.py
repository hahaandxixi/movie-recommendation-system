"""让源码目录在任意克隆目录名下都可作为 ``myidea`` 包导入。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def ensure_source_package(package_name: str = "myidea") -> None:
    source_dir = Path(__file__).resolve().parent

    if source_dir.name == package_name:
        parent = str(source_dir.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        return

    if package_name in sys.modules:
        return

    spec = importlib.util.spec_from_file_location(
        package_name,
        source_dir / "__init__.py",
        submodule_search_locations=[str(source_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {source_dir} 注册源码包 {package_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
