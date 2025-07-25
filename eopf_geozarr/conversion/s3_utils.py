"""S3 utilities for GeoZarr conversion."""

import json
import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import s3fs
import zarr
from zarr.storage import FsspecStore


def is_s3_path(path: str) -> bool:
    """
    Check if a path is an S3 URL.

    Parameters
    ----------
    path : str
        Path to check

    Returns
    -------
    bool
        True if the path is an S3 URL
    """
    return path.startswith("s3://")


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    """
    Parse an S3 path into bucket and key components.

    Parameters
    ----------
    s3_path : str
        S3 path in format s3://bucket/key

    Returns
    -------
    tuple[str, str]
        Tuple of (bucket, key)
    """
    parsed = urlparse(s3_path)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    
    return bucket, key


def get_s3_storage_options(s3_path: str, **s3_kwargs) -> Dict[str, Any]:
    """
    Get storage options for S3 access with xarray.

    Parameters
    ----------
    s3_path : str
        S3 path in format s3://bucket/key
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem

    Returns
    -------
    Dict[str, Any]
        Storage options dictionary for xarray
    """
    # Set up default S3 configuration
    default_s3_kwargs = {
        "anon": False,  # Use credentials
        "use_ssl": True,
        "client_kwargs": {
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        }
    }
    
    # Add custom endpoint support (e.g., for OVH Cloud)
    if "AWS_S3_ENDPOINT" in os.environ:
        default_s3_kwargs["endpoint_url"] = os.environ["AWS_S3_ENDPOINT"]
        default_s3_kwargs["client_kwargs"]["endpoint_url"] = os.environ["AWS_S3_ENDPOINT"]
    
    # Merge with user-provided kwargs
    s3_config = {**default_s3_kwargs, **s3_kwargs}
    
    return s3_config


def create_s3_store(s3_path: str, **s3_kwargs) -> str:
    """
    Create an S3 path with storage options for Zarr operations.
    
    This function now returns the S3 path directly, to be used with
    xarray's storage_options parameter instead of creating a store.

    Parameters
    ----------
    s3_path : str
        S3 path in format s3://bucket/key
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem

    Returns
    -------
    str
        S3 path to be used with storage_options
    """
    # Just return the S3 path - storage options will be handled separately
    return s3_path


def write_s3_json_metadata(s3_path: str, metadata: Dict[str, Any], **s3_kwargs) -> None:
    """
    Write JSON metadata directly to S3.

    This is used for writing zarr.json files and other metadata that need
    to be written directly to S3 without going through the Zarr store.

    Parameters
    ----------
    s3_path : str
        S3 path for the JSON file
    metadata : dict
        Metadata dictionary to write as JSON
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem
    """
    # Set up default S3 configuration
    default_s3_kwargs = {
        "anon": False,
        "use_ssl": True,
        "asynchronous": False,  # Force synchronous mode
        "client_kwargs": {
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        }
    }
    
    # Add custom endpoint support (e.g., for OVH Cloud)
    if "AWS_S3_ENDPOINT" in os.environ:
        default_s3_kwargs["endpoint_url"] = os.environ["AWS_S3_ENDPOINT"]
        default_s3_kwargs["client_kwargs"]["endpoint_url"] = os.environ["AWS_S3_ENDPOINT"]
    
    s3_config = {**default_s3_kwargs, **s3_kwargs}
    fs = s3fs.S3FileSystem(**s3_config)
    
    # Write JSON content
    json_content = json.dumps(metadata, indent=2)
    with fs.open(s3_path, "w") as f:
        f.write(json_content)


def s3_path_exists(s3_path: str, **s3_kwargs) -> bool:
    """
    Check if an S3 path exists.

    Parameters
    ----------
    s3_path : str
        S3 path to check
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem

    Returns
    -------
    bool
        True if the path exists
    """
    default_s3_kwargs = {
        "anon": False,
        "use_ssl": True,
        "asynchronous": False,  # Force synchronous mode
        "client_kwargs": {
            "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        }
    }
    
    # Add custom endpoint support (e.g., for OVH Cloud)
    if "AWS_S3_ENDPOINT" in os.environ:
        default_s3_kwargs["endpoint_url"] = os.environ["AWS_S3_ENDPOINT"]
        default_s3_kwargs["client_kwargs"]["endpoint_url"] = os.environ["AWS_S3_ENDPOINT"]
    
    s3_config = {**default_s3_kwargs, **s3_kwargs}
    fs = s3fs.S3FileSystem(**s3_config)
    
    return fs.exists(s3_path)


def open_s3_zarr_group(s3_path: str, mode: str = "r", **s3_kwargs) -> zarr.Group:
    """
    Open a Zarr group from S3.

    Parameters
    ----------
    s3_path : str
        S3 path to the Zarr group
    mode : str, default "r"
        Access mode ("r", "r+", "w", "a")
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem

    Returns
    -------
    zarr.Group
        Zarr group
    """
    store = create_s3_store(s3_path, **s3_kwargs)
    return zarr.open_group(store, mode=mode, zarr_format=3)


def get_s3_credentials_info() -> Dict[str, Optional[str]]:
    """
    Get information about available S3 credentials.

    Returns
    -------
    dict
        Dictionary with credential information
    """
    return {
        "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": "***" if os.environ.get("AWS_SECRET_ACCESS_KEY") else None,
        "aws_session_token": "***" if os.environ.get("AWS_SESSION_TOKEN") else None,
        "aws_default_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "aws_profile": os.environ.get("AWS_PROFILE"),
        "aws_s3_endpoint": os.environ.get("AWS_S3_ENDPOINT"),
    }


def validate_s3_access(s3_path: str, **s3_kwargs) -> tuple[bool, Optional[str]]:
    """
    Validate that we can access the S3 path.

    Parameters
    ----------
    s3_path : str
        S3 path to validate
    **s3_kwargs
        Additional keyword arguments for s3fs.S3FileSystem

    Returns
    -------
    tuple[bool, Optional[str]]
        Tuple of (success, error_message)
    """
    try:
        bucket, key = parse_s3_path(s3_path)
        
        default_s3_kwargs = {
            "anon": False,
            "use_ssl": True,
            "asynchronous": False,  # Force synchronous mode
            "client_kwargs": {
                "region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            }
        }
        
        # Add custom endpoint support (e.g., for OVH Cloud)
        if "AWS_S3_ENDPOINT" in os.environ:
            default_s3_kwargs["endpoint_url"] = os.environ["AWS_S3_ENDPOINT"]
            default_s3_kwargs["client_kwargs"]["endpoint_url"] = os.environ["AWS_S3_ENDPOINT"]
        
        s3_config = {**default_s3_kwargs, **s3_kwargs}
        fs = s3fs.S3FileSystem(**s3_config)
        
        # Try to list the bucket to check access
        fs.ls(f"s3://{bucket}", detail=False)
        
        return True, None
        
    except Exception as e:
        return False, str(e)
