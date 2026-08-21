"""
Object storage client supporting Cloudflare R2, AWS S3, and local filesystem fallback.
"""

import io
import os
import shutil
from pathlib import Path
from typing import Optional
from .settings import settings


class StorageClient:
    """
    Object storage interface for audio files, dataset manifests, and model checkpoints.
    Uses local filesystem in dev/test if S3 credentials are mock/unset.
    """

    def __init__(self, local_root: str = "data/storage"):
        self.local_root = Path(local_root)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.use_local = (
            settings.S3_ACCESS_KEY_ID == "mock-access-key" or 
            settings.S3_ENDPOINT_URL is None or 
            "localhost" in (settings.S3_ENDPOINT_URL or "")
        )

    async def put_object(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload an object and return its storage URI (s3:// or file://)."""
        if self.use_local:
            target_path = self.local_root / key
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "wb") as f:
                f.write(data)
            return f"s3://{settings.S3_BUCKET_NAME}/{key}"
        else:
            # S3 / R2 Boto3 client implementation
            import boto3
            session = boto3.session.Session()
            client = session.client(
                's3',
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.S3_ACCESS_KEY_ID,
                aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
                region_name=settings.S3_REGION_NAME
            )
            client.put_object(
                Bucket=settings.S3_BUCKET_NAME,
                Key=key,
                Body=data,
                ContentType=content_type
            )
            return f"s3://{settings.S3_BUCKET_NAME}/{key}"

    async def get_object(self, key: str) -> bytes:
        """Retrieve object bytes by key."""
        if self.use_local:
            clean_key = key.replace(f"s3://{settings.S3_BUCKET_NAME}/", "")
            target_path = self.local_root / clean_key
            if not target_path.exists():
                raise FileNotFoundError(f"Object not found in local storage: {clean_key}")
            with open(target_path, "rb") as f:
                return f.read()
        else:
            import boto3
            clean_key = key.replace(f"s3://{settings.S3_BUCKET_NAME}/", "")
            session = boto3.session.Session()
            client = session.client(
                's3',
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.S3_ACCESS_KEY_ID,
                aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
                region_name=settings.S3_REGION_NAME
            )
            resp = client.get_object(Bucket=settings.S3_BUCKET_NAME, Key=clean_key)
            return resp['Body'].read()

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a short-lived presigned download URL."""
        clean_key = key.replace(f"s3://{settings.S3_BUCKET_NAME}/", "")
        if self.use_local:
            return f"/v1/audio/download/{clean_key}"
        else:
            import boto3
            session = boto3.session.Session()
            client = session.client(
                's3',
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.S3_ACCESS_KEY_ID,
                aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
                region_name=settings.S3_REGION_NAME
            )
            return client.generate_presigned_url(
                'get_object',
                Params={'Bucket': settings.S3_BUCKET_NAME, 'Key': clean_key},
                ExpiresIn=expires_in
            )


storage = StorageClient()
