from pathlib import Path

import urllib3
from minio import Minio


class ObjectStorage:
    def __init__(self, endpoint: str, access_key: str, secret_key: str, bucket: str, secure: bool = False) -> None:
        http_client = urllib3.PoolManager(timeout=urllib3.Timeout(connect=1.0, read=2.0), retries=False)
        self.client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure, http_client=http_client
        )
        self.bucket = bucket

    def ready(self) -> bool:
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
            return True
        except Exception:
            return False

    def ensure_bucket(self) -> None:
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload(self, path: Path, object_name: str, content_type: str) -> str:
        self.ensure_bucket()
        self.client.fput_object(self.bucket, object_name, str(path), content_type=content_type)
        return object_name

    def download(self, object_name: str, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.fget_object(self.bucket, object_name, str(destination))
        return destination
