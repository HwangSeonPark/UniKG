from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType

_BASE = Path(__file__).resolve().parents[1]
_MODULE_CACHE: dict[str, ModuleType] = {}


def _ensure_package(folder: str) -> str:
    pkg_name = "evaluate.construction"
    if not folder:
        return pkg_name

    parts = folder.split("/")
    package_parts = [p.replace("-", "_") for p in parts]
    current_dir = _BASE

    for physical, segment in zip(parts, package_parts):
        pkg_name = f"{pkg_name}.{segment}"
        if pkg_name not in sys.modules:
            module = ModuleType(pkg_name)
            module.__path__ = [str(current_dir / physical)]  # type: ignore[attr-defined]
            sys.modules[pkg_name] = module
        current_dir = current_dir / physical

    return pkg_name


def load_module(folder: str, module_name: str) -> ModuleType:
    """
    폴더명에 하이픈이 포함되어도 안전하게 단일 파이썬 모듈로 로드한다.
    """
    key = f"{folder}:{module_name}"
    if key in _MODULE_CACHE:
        return _MODULE_CACHE[key]

    target = _BASE / folder / f"{module_name}.py"
    if not target.exists():
        raise FileNotFoundError(f"모듈을 찾을 수 없습니다: {target}")

    package_name = _ensure_package(folder)
    qualified_name = f"{package_name}.{module_name}"

    spec = spec_from_file_location(qualified_name, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"{qualified_name} 모듈을 로딩할 수 없습니다.")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[call-arg]
    _MODULE_CACHE[key] = module
    return module


