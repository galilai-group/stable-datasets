"""Helpers for cache locations that are local paths or URIs.

A cache location is a local path or a URI such as ``s3://bucket/prefix``
or ``file:///data/cache``. Keep it as a ``str``: ``pathlib.Path`` changes
``s3://bucket`` to ``s3:/bucket``.

Small JSON files (cache metadata, split manifests) use ``pyarrow.fs``.
The S3 and Azure keys in ``storage_options`` go to the ``pyarrow.fs``
filesystem. Other stores, such as ``gs://``, use the credentials of the
environment for these files. Lance datasets use Lance with the same
``storage_options``.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import pyarrow.fs as pafs


# Lance (object_store) option names and the S3FileSystem arguments they set.
_S3_OPTIONS = {
    "aws_access_key_id": "access_key",
    "access_key_id": "access_key",
    "aws_secret_access_key": "secret_key",
    "secret_access_key": "secret_key",
    "aws_session_token": "session_token",
    "session_token": "session_token",
    "aws_region": "region",
    "region": "region",
    "aws_endpoint": "endpoint_override",
    "endpoint": "endpoint_override",
}

# Lance (object_store) option names and the AzureFileSystem arguments they set.
_AZURE_OPTIONS = {
    "azure_storage_account_name": "account_name",
    "account_name": "account_name",
    "azure_storage_account_key": "account_key",
    "account_key": "account_key",
    "access_key": "account_key",
    "azure_storage_sas_key": "sas_token",
    "sas_key": "sas_token",
    "sas_token": "sas_token",
}


def local_path(uri: str | os.PathLike) -> Path | None:
    """Return the local path of a location, or ``None`` for a remote URI."""
    uri = os.fspath(uri)
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    # A one-letter scheme is a Windows drive letter.
    if len(parsed.scheme) > 1:
        return None
    return Path(uri)


def join(uri: str | os.PathLike, *parts: str) -> str:
    """Join path parts to a location and keep the URI scheme."""
    uri = os.fspath(uri)
    if "://" in uri:
        return "/".join([uri.rstrip("/"), *parts])
    return os.path.join(uri, *parts)


def exists(uri: str, storage_options: dict | None = None) -> bool:
    """Return True if a file or directory exists at the location."""
    fs, path = _filesystem(uri, storage_options)
    return fs.get_file_info(path).type != pafs.FileType.NotFound


def read_text(uri: str, storage_options: dict | None = None) -> str:
    """Read a small UTF-8 text file."""
    fs, path = _filesystem(uri, storage_options)
    with fs.open_input_stream(path) as stream:
        return stream.read().decode("utf-8")


def write_text(uri: str, text: str, storage_options: dict | None = None) -> None:
    """Write a small text file. Replace a local file atomically."""
    path = local_path(uri)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
        return
    fs, remote_path = _filesystem(uri, storage_options)
    with fs.open_output_stream(remote_path) as stream:
        stream.write(text.encode("utf-8"))


def _filesystem(uri: str, storage_options: dict | None) -> tuple[pafs.FileSystem, str]:
    """Return the ``pyarrow.fs`` filesystem and the path for a location."""
    path = local_path(uri)
    if path is not None:
        return pafs.LocalFileSystem(), str(path.absolute())
    parsed = urlparse(uri)
    bucket_path = parsed.netloc + parsed.path
    s3_kwargs = _filesystem_kwargs(storage_options, _S3_OPTIONS)
    if parsed.scheme in ("s3", "s3a") and s3_kwargs:
        endpoint = s3_kwargs.get("endpoint_override")
        if endpoint is not None and "://" in endpoint:
            s3_kwargs["scheme"], s3_kwargs["endpoint_override"] = endpoint.split("://", 1)
        return pafs.S3FileSystem(**s3_kwargs), bucket_path
    if parsed.scheme == "az":
        azure_kwargs = _filesystem_kwargs(storage_options, _AZURE_OPTIONS)
        if "account_name" not in azure_kwargs:
            raise ValueError(f"Give the Azure account name in storage_options to use {uri}.")
        return pafs.AzureFileSystem(**azure_kwargs), bucket_path
    return pafs.FileSystem.from_uri(uri)


def _filesystem_kwargs(storage_options: dict | None, names: dict[str, str]) -> dict:
    return {names[key]: value for key, value in (storage_options or {}).items() if key in names}
