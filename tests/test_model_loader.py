"""Tests for artifact loading."""
import json

import pytest

from app.services.model_loader import ArtifactError, load_artifacts
from tests.conftest import FEATURES, build_model, write_artifacts


def test_loads_synthetic_artifacts(art):
    assert art.feature_columns == FEATURES
    assert art.metrics["model"] == "Synthetic test model"
    assert "AMT_CREDIT" in art.numeric_features
    assert art.categorical_features == ["CODE_GENDER", "NAME_CONTRACT_TYPE"]
    assert set(art.known_categories["CODE_GENDER"]) == {"F", "M"}


def test_missing_artifacts_raise_clear_error(tmp_path):
    with pytest.raises(ArtifactError, match="Missing artifact"):
        load_artifacts(str(tmp_path))


def test_missing_single_file_is_named(tmp_path):
    write_artifacts(tmp_path)
    (tmp_path / "feature_columns.json").unlink()
    with pytest.raises(ArtifactError, match="feature_columns.json"):
        load_artifacts(str(tmp_path))


def test_schema_mismatch_is_rejected(tmp_path):
    write_artifacts(tmp_path, model=build_model(), features=list(reversed(FEATURES)))
    with pytest.raises(ArtifactError, match="does not match"):
        load_artifacts(str(tmp_path))


def test_corrupt_json_is_rejected(tmp_path):
    write_artifacts(tmp_path)
    (tmp_path / "model_metrics.json").write_text("{not valid json")
    with pytest.raises(ArtifactError, match="model_metrics.json"):
        load_artifacts(str(tmp_path))


def test_corrupt_model_file_is_rejected(tmp_path):
    write_artifacts(tmp_path)
    (tmp_path / "credence_risk_model.joblib").write_bytes(b"not a model")
    with pytest.raises(ArtifactError, match="could not be loaded"):
        load_artifacts(str(tmp_path))
    assert json.loads((tmp_path / "feature_columns.json").read_text()) == FEATURES