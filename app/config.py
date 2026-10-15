"""Central configuration. Values come from environment variables or .env, with safe defaults."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def _path(name: str, default: str) -> Path:
    p = Path(os.getenv(name, default))
    return p if p.is_absolute() else ROOT / p


def _thresholds() -> tuple:
    raw = os.getenv("RISK_BAND_THRESHOLDS", "0.10,0.30,0.50")
    msg = ("RISK_BAND_THRESHOLDS must be three increasing numbers between 0 and 1, "
           f"for example 0.10,0.30,0.50 (got: {raw!r})")
    try:
        vals = tuple(float(x) for x in raw.split(","))
    except ValueError as exc:
        raise ValueError(msg) from exc
    if len(vals) != 3 or list(vals) != sorted(vals) or not all(0 < v < 1 for v in vals):
        raise ValueError(msg)
    return vals


APP_TITLE = os.getenv("APP_TITLE", "Credence Risk")
APP_ENV = os.getenv("APP_ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

MODELS_DIR = _path("MODELS_DIR", "models")
REPORTS_DIR = _path("REPORTS_DIR", "reports")

# Illustrative demonstration bands - not calibrated lending grades or approved policy.
RISK_BAND_THRESHOLDS = _thresholds()
RISK_BAND_LABELS = ("Low", "Moderate", "Elevated", "High")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_ROWS = int(os.getenv("MAX_UPLOAD_ROWS", "50000"))

# scikit-learn version that saved the model (reported by the version warning).
EXPECTED_SKLEARN_VERSION = "1.6.1"