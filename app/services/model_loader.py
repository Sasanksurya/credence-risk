"""Loads the saved model, feature schema and metrics. Never trains or replaces a model."""
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

import joblib
import sklearn

from app.config import EXPECTED_SKLEARN_VERSION, MODELS_DIR

logger = logging.getLogger(__name__)

MODEL_FILE = "credence_risk_model.joblib"
METRICS_FILE = "model_metrics.json"
FEATURES_FILE = "feature_columns.json"


class ArtifactError(RuntimeError):
    """Raised when a model artifact is missing or incompatible."""


@dataclass
class Artifacts:
    model: object
    feature_columns: list
    metrics: dict
    numeric_features: list
    categorical_features: list
    known_categories: dict = field(default_factory=dict)
    version_warning: Optional[str] = None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Could not read {path.name}: {exc}") from exc


@lru_cache(maxsize=2)
def load_artifacts(models_dir: Optional[str] = None) -> Artifacts:
    """Load artifacts once per process. Only load model files you created yourself."""
    folder = Path(models_dir) if models_dir else MODELS_DIR
    required = [MODEL_FILE, METRICS_FILE, FEATURES_FILE]
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise ArtifactError(
            f"Missing artifact file(s) in {folder}: {', '.join(missing)}. "
            "Generate them from the training notebook; the app never trains a replacement."
        )

    try:
        model = joblib.load(folder / MODEL_FILE)
    except Exception as exc:  # incompatible versions surface as several error types
        raise ArtifactError(
            f"The model could not be loaded (installed scikit-learn {sklearn.__version__}, "
            f"model saved with {EXPECTED_SKLEARN_VERSION}): {exc}"
        ) from exc

    feature_columns = _read_json(folder / FEATURES_FILE)
    metrics = _read_json(folder / METRICS_FILE)

    try:
        pre = model.named_steps["preprocessor"]
        numeric, categorical = [], []
        for name, _, cols in pre.transformers_:
            if name == "num":
                numeric = list(cols)
            elif name == "cat":
                categorical = list(cols)
        encoder = pre.named_transformers_["cat"].named_steps["encoder"]
        known = {c: [str(v) for v in cats] for c, cats in zip(categorical, encoder.categories_)}
    except Exception as exc:
        raise ArtifactError(f"Unexpected model structure: {exc}") from exc

    if list(getattr(model, "feature_names_in_", [])) != list(feature_columns):
        raise ArtifactError("feature_columns.json does not match the columns stored in the model.")

    warning = None
    if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
        warning = (f"scikit-learn {sklearn.__version__} is installed but the model was saved with "
                   f"{EXPECTED_SKLEARN_VERSION}. Predictions may be unreliable.")
        logger.warning(warning)

    return Artifacts(model, list(feature_columns), metrics, numeric, categorical, known, warning)