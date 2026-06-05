import os
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.calibration import CalibratedClassifierCV

ALGORITHM_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "SVM (Linear)",
    "Naive Bayes",
]


def _build_pipeline(classifier_name: str) -> Pipeline:
    """Build sklearn Pipeline for the chosen algorithm."""
    if classifier_name == "Logistic Regression":
        return Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ])
    elif classifier_name == "Random Forest":
        return Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ])
    elif classifier_name == "SVM (Linear)":
        # Wrap with CalibratedClassifierCV so predict_proba is available
        base = LinearSVC(max_iter=3000, C=1.0, random_state=42)
        calibrated = CalibratedClassifierCV(base, cv=3)
        return Pipeline([
            ("vec", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
            ("clf", calibrated),
        ])
    elif classifier_name == "Naive Bayes":
        # CountVectorizer produces non-negative integer features required by MultinomialNB
        return Pipeline([
            ("vec", CountVectorizer(ngram_range=(1, 2), max_features=10000)),
            ("clf", MultinomialNB(alpha=0.5)),
        ])
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")


class InsuranceClassifier:
    def __init__(self, classifier_name: str = "Logistic Regression"):
        if classifier_name not in ALGORITHM_NAMES:
            raise ValueError(f"classifier_name must be one of {ALGORITHM_NAMES}")
        self.classifier_name = classifier_name
        self.pipeline: Pipeline | None = None
        self.is_trained: bool = False
        self.training_accuracy: float | None = None
        self.cv_scores: np.ndarray | None = None
        self.classes_: list[str] = []
        self.n_samples_trained: int = 0

    def train(self, texts: list[str], compound_labels: list[str]) -> None:
        if len(texts) < 2:
            raise ValueError("Need at least 2 training samples.")

        self.pipeline = _build_pipeline(self.classifier_name)
        self.pipeline.fit(texts, compound_labels)

        self.is_trained = True
        self.classes_ = list(self.pipeline.classes_)
        self.n_samples_trained = len(texts)

        n_classes = len(set(compound_labels))
        # Use cross-validation only when we have enough samples per class
        min_class_count = min(
            sum(1 for l in compound_labels if l == c) for c in set(compound_labels)
        )
        cv_folds = min(5, min_class_count) if min_class_count >= 2 else None

        if cv_folds and cv_folds >= 2 and len(texts) >= cv_folds:
            self.cv_scores = cross_val_score(
                _build_pipeline(self.classifier_name),
                texts, compound_labels,
                cv=cv_folds, scoring="accuracy",
            )
            self.training_accuracy = float(self.cv_scores.mean())
        else:
            preds = self.pipeline.predict(texts)
            correct = sum(p == l for p, l in zip(preds, compound_labels))
            self.training_accuracy = correct / len(compound_labels)

    def predict(self, text: str) -> dict:
        self._require_trained()
        compound = self.pipeline.predict([text])[0]
        confidence = self._get_confidence(text)
        return self._parse_result(compound, confidence)

    def predict_top_n(self, text: str, n: int = 3) -> list[dict]:
        self._require_trained()
        clf = self.pipeline.named_steps["clf"]

        if hasattr(clf, "predict_proba"):
            proba = self.pipeline.predict_proba([text])[0]
            top_idx = np.argsort(proba)[::-1][:n]
            return [
                self._parse_result(self.pipeline.classes_[i], float(proba[i]))
                for i in top_idx
                if float(proba[i]) > 0.0
            ]

        # Fallback: return only the top prediction
        return [self.predict(text)]

    def _get_confidence(self, text: str) -> float | None:
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            proba = self.pipeline.predict_proba([text])[0]
            return float(proba.max())
        return None

    @staticmethod
    def _parse_result(compound: str, confidence: float | None) -> dict:
        compound = str(compound)
        parts = compound.split("|", 1)
        return {
            "category": parts[0],
            "subcategory": parts[1] if len(parts) > 1 else "",
            "compound_label": compound,
            "confidence": confidence,
        }

    def _require_trained(self) -> None:
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model is not trained yet. Train first.")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "pipeline": self.pipeline,
            "classifier_name": self.classifier_name,
            "is_trained": self.is_trained,
            "training_accuracy": self.training_accuracy,
            "cv_scores": self.cv_scores,
            "classes_": self.classes_,
            "n_samples_trained": self.n_samples_trained,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str) -> "InsuranceClassifier":
        payload = joblib.load(path)
        obj = cls(classifier_name=payload["classifier_name"])
        obj.pipeline = payload["pipeline"]
        obj.is_trained = payload["is_trained"]
        obj.training_accuracy = payload["training_accuracy"]
        obj.cv_scores = payload.get("cv_scores")
        obj.classes_ = payload.get("classes_", [])
        obj.n_samples_trained = payload.get("n_samples_trained", 0)
        return obj
