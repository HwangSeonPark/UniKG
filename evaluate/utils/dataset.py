from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = PROJECT_ROOT / "evaluate" / "references"


def _ensure_path(path: Path, label: str) -> Path:
    if not path.exists():
        parent = path.parent
        if parent.exists():
            available = [f.name for f in parent.iterdir() if f.is_file()]
            raise FileNotFoundError(
                f"{label} path not found: {path}\n"
                f"Available files in {parent}: {', '.join(sorted(available))}"
            )
        raise FileNotFoundError(f"{label} path not found: {path}")
    return path


def resolve_dataset_paths(
    dataset_dir: Optional[str],
    pred_path: Optional[str],
    gold_path: Optional[str],
    text_path: Optional[str],
    require_text: bool,
    require_gold: bool = True,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Resolve actual file paths from a dataset directory and per-file arguments.
    - pred_path is required
    - gold_path is required when require_gold is True
    - text_path is required when require_text is True
    - if dataset_dir is provided, relative filenames are resolved from it
    """

    base = Path(dataset_dir).expanduser().resolve() if dataset_dir else None

    def _resolve(candidate: Optional[str], label: str, required: bool = True) -> Optional[Path]:
        if not candidate:
            if required:
                if base:
                    available = [f.name for f in base.iterdir() if f.is_file()]
                    raise ValueError(
                        f"{label} path is required. "
                        f"Use --{label} to specify. "
                        f"Available files in {base}: {', '.join(sorted(available))}"
                    )
                raise ValueError(f"{label} path is required. Use --{label} to specify.")
            return None
        
        candidate_path = Path(candidate).expanduser()
        if candidate_path.is_absolute():
            return _ensure_path(candidate_path.resolve(), label)
        
        if base:
            return _ensure_path((base / candidate).resolve(), label)
        
        return _ensure_path(candidate_path.resolve(), label)

    pred = _resolve(pred_path, "pred", required=True)
    gold = _resolve(gold_path, "gold", required=require_gold)

    text: Optional[Path] = None
    if require_text or text_path:
        try:
            text = _resolve(text_path, "text", required=require_text)
        except (ValueError, FileNotFoundError) as e:
            if require_text:
                raise
            text = None

    return str(pred), str(gold) if gold else None, str(text) if text else None


def infer_dataset_dir(explicit: Optional[str]) -> str:
    """
    if dataset directory is not specified, use the internal reference as the default value.
    """
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    return str(DEFAULT_DATASET_DIR)


