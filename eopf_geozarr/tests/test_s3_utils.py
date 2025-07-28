"""Tests for S3 utilities."""

from unittest.mock import Mock, patch

import pytest

from eopf_geozarr.conversion.s3_utils import (
    create_s3_store,
    get_s3_credentials_info,
    get_s3_storage_options,
    is_s3_path,
    parse_s3_path,
    validate_s3_access,
)


def test_is_s3_path():
    """Test S3 path detection."""
    assert is_s3_path("s3://bucket/path")
    assert is_s3_path("s3://my-bucket/data/file.zarr")
    assert not is_s3_path("/local/path")
    assert not is_s3_path("https://example.com")
    assert not is_s3_path("gs://bucket/path")


def test_parse_s3_path():
    """Test S3 path parsing."""
    bucket, key = parse_s3_path("s3://my-bucket/data/file.zarr")
    assert bucket == "my-bucket"
    assert key == "data/file.zarr"

    bucket, key = parse_s3_path("s3://bucket/")
    assert bucket == "bucket"
    assert key == ""

    bucket, key = parse_s3_path("s3://bucket/single-file")
    assert bucket == "bucket"
    assert key == "single-file"

    with pytest.raises(ValueError):
        parse_s3_path("https://example.com")


def test_get_s3_credentials_info():
    """Test S3 credentials info retrieval."""
    with patch.dict(
        "os.environ",
        {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_DEFAULT_REGION": "us-west-2",
        },
    ):
        creds = get_s3_credentials_info()
        assert creds["aws_access_key_id"] == "test-key"
        assert creds["aws_secret_access_key"] == "***"
        assert creds["aws_default_region"] == "us-west-2"


@patch("eopf_geozarr.conversion.s3_utils.s3fs.S3FileSystem")
def test_validate_s3_access_success(mock_s3fs):
    """Test successful S3 access validation."""
    mock_fs = Mock()
    mock_fs.ls.return_value = ["file1", "file2"]
    mock_s3fs.return_value = mock_fs

    success, error = validate_s3_access("s3://test-bucket/path")
    assert success is True
    assert error is None
    mock_fs.ls.assert_called_once_with("s3://test-bucket", detail=False)


@patch("eopf_geozarr.conversion.s3_utils.s3fs.S3FileSystem")
def test_validate_s3_access_failure(mock_s3fs):
    """Test failed S3 access validation."""
    mock_fs = Mock()
    mock_fs.ls.side_effect = Exception("Access denied")
    mock_s3fs.return_value = mock_fs

    success, error = validate_s3_access("s3://test-bucket/path")
    assert success is False
    assert "Access denied" in error


def test_create_s3_store_path_handling():
    """Test that create_s3_store returns the S3 path correctly."""
    # Test with S3 path
    result = create_s3_store("s3://test-bucket/path/to/data")
    assert result == "s3://test-bucket/path/to/data"

    # Test with bucket only
    result = create_s3_store("s3://test-bucket")
    assert result == "s3://test-bucket"


def test_get_s3_storage_options():
    """Test that get_s3_storage_options returns correct configuration."""
    with patch.dict(
        "os.environ",
        {"AWS_DEFAULT_REGION": "us-west-2", "AWS_S3_ENDPOINT": "https://s3.example.com"},
    ):
        options = get_s3_storage_options("s3://test-bucket/path")

        assert options["anon"] is False
        assert options["use_ssl"] is True
        assert options["client_kwargs"]["region_name"] == "us-west-2"
        assert options["endpoint_url"] == "https://s3.example.com"
        assert options["client_kwargs"]["endpoint_url"] == "https://s3.example.com"
