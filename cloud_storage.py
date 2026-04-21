# cloud_storage.py
import boto3
import io
from pathlib import Path
from typing import Optional

class R2Storage:
    """Cloudflare R2 storage handler (S3-compatible)"""
    
    def __init__(self, endpoint_url: str, access_key: str, 
                 secret_key: str, bucket_name: str):
        self.client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='auto'  # R2 uses 'auto' for region
        )
        self.bucket = bucket_name
    
    def upload_file(self, local_path: Path, remote_key: str) -> Optional[str]:
        """Upload a file to R2 and return the public URL"""
        if not local_path.exists():
            return None
        
        self.client.upload_file(
            str(local_path),
            self.bucket,
            remote_key,
            ExtraArgs={'ContentType': self._get_content_type(local_path)}
        )
        
        # R2 public URL format
        return f"https://pub-{self.bucket}.r2.dev/{remote_key}"
    
    def _get_content_type(self, path: Path) -> str:
        suffixes = {
            '.html': 'text/html',
            '.png': 'image/png',
            '.json': 'application/json',
            '.csv': 'text/csv',
        }
        return suffixes.get(path.suffix, 'application/octet-stream')