"""Tests for CSV upload validation in the Batch Scoring page."""
import io

import pytest

from app.views import batch


class FakeUpload(io.BytesIO):
    """Minimal stand-in for Streamlit's UploadedFile."""

    def __init__(self, data: bytes):
        super().__init__(data)
        self.size = len(data)


def test_valid_csv_is_read():
    df = batch.read_upload(FakeUpload(b"a,b\n1,2\n3,4\n"))
    assert df.shape == (2, 2)


def test_empty_file_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        batch.read_upload(FakeUpload(b""))


def test_header_only_file_is_rejected():
    with pytest.raises(ValueError, match="no data rows"):
        batch.read_upload(FakeUpload(b"a,b\n"))


def test_invalid_encoding_is_rejected():
    with pytest.raises(ValueError, match="UTF-8"):
        batch.read_upload(FakeUpload(b"a,b\n\xff\xfe,2\n"))


def test_row_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(batch, "MAX_UPLOAD_ROWS", 2)
    with pytest.raises(ValueError, match="more than 2 rows"):
        batch.read_upload(FakeUpload(b"a\n1\n2\n3\n"))


def test_size_limit_is_enforced(monkeypatch):
    monkeypatch.setattr(batch, "MAX_UPLOAD_MB", 0)
    with pytest.raises(ValueError, match="larger than"):
        batch.read_upload(FakeUpload(b"a\n1\n"))