import json
import uuid
from datetime import datetime
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data"
_TRAINING_FILE = _DATA_DIR / "training_data.json"


def _ensure_dir() -> None:
    _DATA_DIR.mkdir(exist_ok=True)


def load_training_data() -> list[dict]:
    _ensure_dir()
    if not _TRAINING_FILE.exists():
        return []
    with open(_TRAINING_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_training_data(samples: list[dict]) -> None:
    _ensure_dir()
    with open(_TRAINING_FILE, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)


def add_training_sample(
    text: str,
    filename: str,
    category: str,
    subcategory: str,
    keywords: list[str],
) -> dict:
    samples = load_training_data()
    sample = {
        "id": str(uuid.uuid4()),
        "filename": filename,
        "text": text,
        "category": category.strip(),
        "subcategory": subcategory.strip(),
        "keywords": [k.strip() for k in keywords if k.strip()],
        "compound_label": f"{category.strip()}|{subcategory.strip()}",
        "timestamp": datetime.now().isoformat(),
    }
    samples.append(sample)
    _write_training_data(samples)
    return sample


def delete_training_sample(sample_id: str) -> None:
    samples = load_training_data()
    samples = [s for s in samples if s["id"] != sample_id]
    _write_training_data(samples)


def clear_all_training_data() -> None:
    _write_training_data([])


def get_all_categories(include_defaults: bool = True) -> list[str]:
    from .defaults import DEFAULT_CATEGORIES
    samples = load_training_data()
    user_cats = sorted({s["category"] for s in samples})
    if include_defaults:
        return sorted(set(DEFAULT_CATEGORIES) | set(user_cats))
    return user_cats


def get_subcategories_for(category: str, include_defaults: bool = True) -> list[str]:
    from .defaults import DEFAULT_SUBCATEGORIES
    samples = load_training_data()
    user_subcats = sorted({s["subcategory"] for s in samples if s["category"] == category})
    if include_defaults:
        default = DEFAULT_SUBCATEGORIES.get(category, [])
        return sorted(set(default) | set(user_subcats))
    return user_subcats
