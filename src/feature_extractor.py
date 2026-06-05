import re


def preprocess_text(text: str) -> str:
    text = text.lower()
    # Keep alphanumeric and spaces; collapse whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_single_text(text: str, keywords: list | None = None) -> str:
    """Preprocess text and append keywords with boosted weight."""
    processed = preprocess_text(text)
    if keywords:
        # Repeat keywords to give them higher TF weight
        kw_block = " ".join(preprocess_text(k) for k in keywords if k.strip())
        kw_boost = (kw_block + " ") * 3
        return f"{processed} {kw_boost}".strip()
    return processed


def prepare_training_texts(samples: list[dict]) -> list[str]:
    """Prepare feature strings for a list of training samples."""
    return [
        prepare_single_text(s["text"], s.get("keywords", []))
        for s in samples
    ]
