
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

try:  # boto3 is optional at import time.
    import boto3
    from botocore.exceptions import ClientError
    _HAS_BOTO3 = True
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore
    ClientError = Exception  # type: ignore
    _HAS_BOTO3 = False


CONTENT_TYPES = {
    ".html": "text/html",
    ".htm": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".json": "application/json",
    ".csv": "text/csv",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".geojson": "application/geo+json",
    ".txt": "text/plain",
}


def _content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")


class R2Storage:
    """Thin wrapper around the S3 API exposed by Cloudflare R2."""

    def __init__(self, endpoint_url: str, access_key: str,
                 secret_key: str, bucket_name: str,
                 public_base_url: Optional[str] = None) -> None:
        if not _HAS_BOTO3:
            raise RuntimeError("boto3 is not installed; cannot use R2Storage.")
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )
        self.bucket = bucket_name
        # Public base URL for objects. If unset, we construct the default
        # r2.dev URL. For custom domains, set ``public_base_url`` explicitly.
        self.public_base_url = (
            public_base_url.rstrip("/") if public_base_url
            else f"https://pub-{bucket_name}.r2.dev"
        )

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def upload_file(self, local_path: Path, remote_key: str) -> Optional[str]:
        """Upload ``local_path`` to ``remote_key``. Returns the public URL."""
        local_path = Path(local_path)
        if not local_path.exists() or not local_path.is_file():
            return None
        try:
            self.client.upload_file(
                str(local_path),
                self.bucket,
                remote_key,
                ExtraArgs={"ContentType": _content_type(local_path)},
            )
        except ClientError:
            return None
        return self.public_url(remote_key)

    def upload_bytes(self, data: bytes, remote_key: str,
                     content_type: str = "application/octet-stream"
                     ) -> Optional[str]:
        """Upload ``data`` directly; useful for in-memory HTML/JSON."""
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=remote_key,
                Body=data,
                ContentType=content_type,
            )
        except ClientError:
            return None
        return self.public_url(remote_key)

    def download_file(self, remote_key: str, local_path: Path) -> bool:
        """Download ``remote_key`` to ``local_path``. Returns True on success."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, remote_key, str(local_path))
            return True
        except ClientError:
            return False

    def exists(self, remote_key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=remote_key)
            return True
        except ClientError:
            return False

    def list_files(self, prefix: str = "") -> List[str]:
        """Return all object keys matching ``prefix``."""
        keys: List[str] = []
        token: Optional[str] = None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            try:
                resp = self.client.list_objects_v2(**kwargs)
            except ClientError:
                return keys
            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])
            if not resp.get("IsTruncated"):
                break
            token = resp.get("NextContinuationToken")
        return keys

    def public_url(self, remote_key: str) -> str:
        return f"{self.public_base_url}/{remote_key.lstrip('/')}"

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------
    def sync_prefix_to_local(self, prefix: str, local_root: Path) -> int:
        """Download every object under ``prefix`` into ``local_root``.

        Preserves the key's subpath below ``prefix`` as the relative path on
        disk.  Returns the number of files downloaded.
        """
        local_root = Path(local_root)
        count = 0
        prefix_clean = prefix.rstrip("/") + "/" if prefix else ""
        for key in self.list_files(prefix_clean):
            rel = key[len(prefix_clean):] if prefix_clean else key
            if not rel:
                continue
            target = local_root / rel
            if self.download_file(key, target):
                count += 1
        return count

    def sync_local_to_prefix(self, local_root: Path, prefix: str,
                             patterns: Optional[Iterable[str]] = None
                             ) -> List[str]:
        """Upload every file under ``local_root`` under ``prefix``."""
        local_root = Path(local_root)
        uploaded: List[str] = []
        if not local_root.exists():
            return uploaded
        prefix_clean = prefix.rstrip("/")
        pats = list(patterns) if patterns else ["*"]
        for pat in pats:
            for p in local_root.rglob(pat):
                if not p.is_file():
                    continue
                rel = p.relative_to(local_root).as_posix()
                key = f"{prefix_clean}/{rel}" if prefix_clean else rel
                url = self.upload_file(p, key)
                if url:
                    uploaded.append(url)
        return uploaded


# ---------------------------------------------------------------------------
# Streamlit integration helpers
# ---------------------------------------------------------------------------
def get_r2_from_secrets(st_secrets) -> Optional[R2Storage]:
    """Construct an :class:`R2Storage` from Streamlit secrets.

    Expected shape (``.streamlit/secrets.toml``)::

        [r2]
        endpoint_url     = "https://<account>.r2.cloudflarestorage.com"
        access_key       = "..."
        secret_key       = "..."
        bucket_name      = "flood-forecast"
        public_base_url  = "https://pub-<account>.r2.dev"  # optional

    Returns ``None`` if boto3 is missing, no ``[r2]`` section is present, or
    any required field is missing/empty.
    """
    if not _HAS_BOTO3:
        return None
    try:
        section = st_secrets.get("r2") if hasattr(st_secrets, "get") else None
        if section is None:
            return None
        required = ("endpoint_url", "access_key", "secret_key", "bucket_name")
        if not all(section.get(k) for k in required):
            return None
        return R2Storage(
            endpoint_url=section["endpoint_url"],
            access_key=section["access_key"],
            secret_key=section["secret_key"],
            bucket_name=section["bucket_name"],
            public_base_url=section.get("public_base_url") or None,
        )
    except Exception:
        return None


def is_available() -> bool:
    """True if boto3 is importable."""
    return _HAS_BOTO3
